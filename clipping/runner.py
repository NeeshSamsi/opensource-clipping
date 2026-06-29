"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import json
import os

from . import diarization as diarization_mod
from . import engine, metadata, studio, hook_manager


# Standard on-disk cache for the Whisper transcript, written next to the other
# pipeline artifacts in cfg.outputs_dir. Lets a re-run (e.g. after a late-stage
# render crash) skip the slow transcription stage entirely.
TRANSCRIPT_CACHE_FILENAME = "transcript_cache.json"


def _transcript_signature(cfg) -> dict:
    """Identity of a transcript: which video + model + word-grouping produced it.

    If any of these change, the cached transcript no longer matches and must be
    regenerated. Video size (not mtime) is used because the source video is
    re-downloaded each run, which changes mtime but not content/size.
    """
    try:
        video_size = os.path.getsize(cfg.file_video_asli)
    except OSError:
        video_size = 0
    return {
        "video_size": video_size,
        "whisper_model": cfg.whisper_model,
        "max_words_per_subtitle": cfg.max_kata_per_subtitle,
    }


def _load_transcript_cache(path: str, signature: dict):
    """Return (transcript, segments) from *path* if present and matching, else None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("signature") != signature:
        return None
    transkrip = payload.get("transkrip_lengkap")
    segmen = payload.get("data_segmen")
    if not transkrip or not segmen:
        return None
    return transkrip, segmen


def _save_transcript_cache(path: str, signature: dict, transkrip_lengkap: str, data_segmen: list) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signature": signature,
                    "transkrip_lengkap": transkrip_lengkap,
                    "data_segmen": data_segmen,
                },
                f,
                ensure_ascii=False,
            )
    except OSError as e:
        print(f"⚠️ Failed to write transcript cache ({e}); continuing without it.")


def run_pipeline(cfg) -> list[dict]:
    """
    Run the full clipping pipeline:
      1. Download YouTube video
      2. Transcribe with Whisper
      3. Analyse with Gemini AI
      4. Normalize metadata
      5. Prepare glitch transition
      6. Render each clip
      7. Save render_manifest.json

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.

    Returns
    -------
    list[dict]
        Render manifest (one dict per clip).
    """

    # Step 1 — Download
    source_platform = getattr(cfg, "source_platform", "youtube")
    engine.download_video(
        cfg.url_youtube,
        cfg.file_video_asli,
        getattr(cfg, "use_dlp_subs", False),
        getattr(cfg, "download_source_height", "max"),
        source_platform=source_platform,
    )

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []

    import glob

    # Search for any json3 file (since the language can be .id.json3 or .en.json3)
    json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
    file_json3 = json3_files[0] if json3_files else None

    # Only run YouTube JSON3 subtitle search for YouTube sources
    if source_platform == "youtube":
        if (
            getattr(cfg, "use_dlp_subs", False)
            and file_json3
            and os.path.exists(file_json3)
        ):
            transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
            )
            if transkrip_lengkap and data_segmen:
                print(
                    f"✅ Successfully parsed subtitles from YouTube ({os.path.basename(file_json3)}), skipping the Whisper process."
                )

    if not transkrip_lengkap or not data_segmen:
        # Try the on-disk transcript cache before invoking Whisper, so a re-run
        # (e.g. after fixing a later-stage crash) doesn't repeat transcription.
        cache_path = os.path.join(cfg.outputs_dir, TRANSCRIPT_CACHE_FILENAME)
        signature = _transcript_signature(cfg)
        cached = _load_transcript_cache(cache_path, signature)

        if cached is not None:
            transkrip_lengkap, data_segmen = cached
            print(f"✅ Loaded cached transcript from {cache_path}, skipping Whisper.")
        else:
            transkrip_lengkap, data_segmen = engine.transcribe_video(
                cfg.file_video_asli,
                max_words_per_subtitle=cfg.max_kata_per_subtitle,
                model_size=cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )
            _save_transcript_cache(cache_path, signature, transkrip_lengkap, data_segmen)
            print(f"💾 Transcript cached at {cache_path}")

    # Step 3 — Gemini AI analysis
    gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")
    
    if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
        print(f"\n🔄 [3/3] Loading AI data ({cfg.ai_provider}) from local file: {gemini_output_path}")
        with open(gemini_output_path, "r", encoding="utf-8") as f:
            hasil_json = json.load(f)
    else:
        hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
        
        # Save raw gemini json for future loading/reproduction
        with open(gemini_output_path, "w", encoding="utf-8") as f:
            json.dump(hasil_json, f, indent=4, ensure_ascii=False)
        print(f"💾 Raw AI response saved to: {gemini_output_path}")

    # Step 4 — Metadata normalisation
    hasil_json = metadata.normalize_and_validate(hasil_json)
    metadata.print_preview(hasil_json)

    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 5 — Diarization (split-screen / camera-switch)
    diarization_data = None
    if (
        (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
        or getattr(cfg, "use_camera_switch", False)
    ) and studio._is_vertical_ratio(cfg.pilihan_rasio):
        try:
            mode_label = (
                "Split-Screen"
                if getattr(cfg, "use_split_screen", False)
                else "Camera-Switch"
            )
            print(f"\n🎙️ [{mode_label}] Running speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
            min_spk = None
            max_spk = None

            if str(num_speakers_arg).lower() == "auto":
                max_faces = studio.estimate_speaker_count_from_video(
                    cfg.file_video_asli, cfg
                )
                num_speakers_arg = "auto"
                min_spk = max(1, max_faces)
                max_spk = min_spk + 2
                print(f"   ℹ️ Pyannote instruction: {min_spk} to {max_spk} speakers.")

            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=num_speakers_arg,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ Diarization failed: {e}")
            print("   Falling back to normal render mode (without split-screen).")
            diarization_data = None

    # Step 6 — Video encoder & glitch
    os.environ["OSC_VIDEO_SCALE_ALGO"] = str(
        getattr(cfg, "video_scale_algo", "lanczos")
    )
    
    # Get target dimensions for auto-bitrate calculation
    import cv2
    cap_e = cv2.VideoCapture(cfg.file_video_asli)
    src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_e.release()
    
    target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.pilihan_rasio, source_h=src_h_e)
    video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

    file_glitch_ts = None
    if cfg.use_hook_glitch:
        print("⚙️ Preparing Glitch Transition Video...")
        
        # Get source dimensions for proper glitch scaling
        import cv2
        cap_g = cv2.VideoCapture(cfg.file_video_asli)
        source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_g.release()

        file_glitch_ts = studio.siapkan_glitch_video(
            cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
        )

    # Step 6 — Render each clip
    render_manifest: list[dict] = []

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        print("\n🎣 Downloading custom Hook clip source...")
        custom_hook_path = hook_manager.download_custom_hook(cfg)

    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        
        if custom_hook_path:
            klip["custom_hook_info"] = {"file_path": custom_hook_path}

        hasil_render = studio.proses_klip(
            klip["rank"],
            klip,
            cfg.pilihan_rasio,
            file_glitch_ts,
            data_segmen,
            cfg,
            video_encoder,
            diarization_data=diarization_data,
        )
        if hasil_render:
            render_manifest.append(hasil_render)

    # Step 7 — Save manifest
    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    print(
        f"\n💾 Render manifest saved to {manifest_path} ({len(render_manifest)} items)"
    )
    return render_manifest
