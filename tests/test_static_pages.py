from pathlib import Path


def test_index_page_contains_job_list_and_new_job_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="job-list"' in response.text
    assert 'href="/new-job.html"' in response.text
    assert "/js/app.js" in response.text


def test_index_page_contains_delete_confirmation_modal(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="delete-modal"' in response.text
    assert 'id="confirm-delete-btn"' in response.text
    assert 'id="cancel-delete-btn"' in response.text
    # Delete button must come before Cancel in the DOM (left/right per app.css flex order)
    assert response.text.index('id="confirm-delete-btn"') < response.text.index('id="cancel-delete-btn"')


def test_index_page_contains_jiggle_drag_delete_controls(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="done-button"' in response.text
    assert 'id="trash-target"' in response.text
    assert "jiggle-mode" in response.text
    assert "attachRowGestures" in response.text
    # The old always-visible per-row delete button is gone - press-and-hold
    # + drag-to-trash replaces it entirely, not alongside it.
    assert "job-delete-btn" not in response.text


def test_index_page_suppresses_native_long_press_context_menu(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '"contextmenu"' in response.text
    assert 'draggable="false"' in response.text


def test_new_job_page_only_creates_job_and_redirects_to_job_page(client):
    # Uploading is exclusively job.html's responsibility (the ingestion
    # slide's "add clip" section) so there is a single upload UI in the app,
    # not one on this page and a second one on job.html.
    response = client.get("/new-job.html")
    assert response.status_code == 200
    assert 'id="job-name"' in response.text
    assert 'id="btn-create"' in response.text
    assert "/job.html?id=" in response.text
    assert 'id="file-input"' not in response.text
    assert "local-path" not in response.text


def test_job_page_contains_story_view_and_action_buttons(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="story-bar"' in response.text
    assert 'id="story-viewport"' in response.text
    assert 'id="story-track"' in response.text
    assert 'id="btn-extract"' in response.text
    assert 'id="btn-recommend"' in response.text


def test_frames_page_contains_frame_grid(client):
    response = client.get("/frames.html")
    assert response.status_code == 200
    assert 'id="frame-grid"' in response.text


def test_frames_page_marks_images_loaded_on_load_event(client):
    response = client.get("/frames.html")
    assert response.status_code == 200
    assert "classList.add(\"loaded\")" in response.text
    assert "addEventListener(\"load\"" in response.text


def test_css_disables_mobile_tap_highlight(client):
    response = client.get("/css/app.css")
    assert response.status_code == 200
    assert "-webkit-tap-highlight-color: transparent" in response.text


def test_css_frame_shimmer_respects_reduced_motion(client):
    response = client.get("/css/app.css")
    assert response.status_code == 200
    assert "frame-shimmer" in response.text
    assert "prefers-reduced-motion" in response.text


def test_recommendation_page_served(client):
    response = client.get("/recommendation.html")
    assert response.status_code == 200
    assert 'id="content"' in response.text


def test_job_page_contains_align_button(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="btn-align"' in response.text


def test_alignment_page_served(client):
    response = client.get("/alignment.html")
    assert response.status_code == 200
    assert 'id="content"' in response.text


def test_job_page_contains_extraction_settings_panel(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="extract-fps"' in response.text
    assert 'id="extract-num-frames"' in response.text
    assert 'id="extract-method"' in response.text


def test_job_page_contains_video_info_hint(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="video-info-hint"' in response.text
    assert "/video-info" in response.text


def test_job_page_restarts_polling_after_each_action():
    # Regression test: polling used to be a one-shot pollStages() call that
    # permanently stopped itself once no stage was "running". If a user
    # started a new stage (e.g. clicked "align") after polling had already
    # auto-stopped, the UI would show a single refresh() snapshot and then
    # never update again, even after the background job finished - it would
    # look stuck on "running" forever. The fix: each action button restarts
    # polling by reassigning stopPolling = startPolling() after its apiPost.
    content = (Path(__file__).resolve().parent.parent / "static" / "job.html").read_text(
        encoding="utf-8"
    )
    # 1 initial assignment (`let stopPolling = startPolling()`) + 7 button
    # handlers (add-clip-upload/import-frames/extract/align/recommend/train/
    # export) each reassigning it after apiPost.
    assert content.count("stopPolling = startPolling()") == 8


def test_job_page_contains_story_slides_for_every_stage(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    for stage in ("ingestion", "frame_extraction", "alignment", "training", "export"):
        assert f'data-stage="{stage}"' in response.text
        assert f'id="slide-status-{stage}"' in response.text


def test_job_page_defines_story_navigation_functions(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert "function reachableIndex" in response.text
    assert "function goToSlide" in response.text
    assert "function checkAutoAdvance" in response.text


def test_job_page_uses_custom_select_for_dropdowns(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert "initCustomSelect" in response.text
    assert 'initCustomSelect(document.getElementById("extract-method"))' in response.text
    assert 'initCustomSelect(document.getElementById("export-format"))' in response.text


def test_app_js_defines_custom_select_helper(client):
    response = client.get("/js/app.js")
    assert response.status_code == 200
    assert "function initCustomSelect" in response.text


def test_css_defines_custom_select_styles(client):
    response = client.get("/css/app.css")
    assert response.status_code == 200
    assert ".custom-select-trigger" in response.text
    assert ".custom-select-panel" in response.text
    assert ".custom-select-option" in response.text


def test_job_page_contains_progress_bar_styles_and_render_hook(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert "progress-track" in response.text
    assert "progress-fill" in response.text
    assert "progress-indeterminate" in response.text


def test_job_page_contains_train_button_and_download_link(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="btn-train"' in response.text
    assert 'id="link-training"' in response.text


def test_job_page_contains_log_output_toggle(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert "log-toggle" in response.text
    assert "log-output" in response.text


def test_job_page_contains_usage_panel(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="usage-panel"' in response.text
    assert 'id="usage-cpu"' in response.text
    assert 'id="usage-ram"' in response.text
    assert 'id="usage-gpu"' in response.text
    assert 'id="usage-vram"' in response.text


def test_job_page_contains_variable_exposure_checkbox(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="train-variable-exposure"' in response.text
    assert 'type="checkbox"' in response.text


def test_job_page_contains_import_frames_section(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="import-frames-path"' in response.text
    assert 'id="btn-import-frames"' in response.text


def test_job_page_contains_add_clip_section(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="raw-videos-list"' in response.text
    assert 'id="add-clip-file-input"' in response.text
    assert "multiple" in response.text
    assert 'id="btn-add-clip-upload"' in response.text
    assert 'id="add-clip-upload-list"' in response.text
    assert "add-clip-local-path" not in response.text
    assert "upload-row-error" in response.text


def test_job_page_contains_export_controls(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="export-format"' in response.text
    assert 'id="btn-export"' in response.text
    assert 'id="link-export"' in response.text
    assert 'value="spz"' in response.text
    assert 'value="sog"' in response.text
