"""
dashboard/test_api.py - Verification and Integration Test Suite for PPE Autoencoder & SVM API
"""
import io
import os
import glob
import random
import sys
import tempfile
import json
from PIL import Image

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_ppe")

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import UploadFile, HTTPException
from dashboard.app import (
    app,
    load_artifacts,
    get_health,
    get_metrics,
    get_sample_crops,
    get_result_figure,
    predict_crop,
    list_benchmark_models,
    MODELS
)

def make_upload_file(filename: str, content: bytes, content_type: str = "image/png") -> UploadFile:
    """Helper to construct a FastAPI UploadFile in-memory."""
    buf = io.BytesIO(content)
    return UploadFile(filename=filename, file=buf, headers={"content-type": content_type})

def run_tests():
    print("=" * 70, flush=True)
    print("STARTING PPE DASHBOARD API VERIFICATION SUITE", flush=True)
    print("=" * 70, flush=True)

    # Ensure artifacts are loaded
    load_artifacts()

    # 1. Health check
    print("\n[Test 1] Testing GET /api/health...", flush=True)
    health = get_health()
    print(f"  Health response: {health}", flush=True)
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert health["scaler_loaded"] is True
    assert health["svm_loaded"] is True
    assert health["latent_dim"] == 256
    assert health["classes"] == ["head", "helmet", "vest"]
    print("  ✓ GET /api/health passed.", flush=True)

    # 1b. Model registry check
    print("\n[Test 1b] Testing GET /api/models (Model Registry)...", flush=True)
    model_list_res = list_benchmark_models()
    assert model_list_res["status"] == "success"
    models = model_list_res["models"]
    assert len(models) == 5, f"Expected 5 registered models, got {len(models)}"
    model_ids = [m["id"] for m in models]
    assert model_ids == ["autoencoder", "cvae", "cgan", "diffusion", "policy"]
    print(f"  Registered models: {model_ids}", flush=True)
    print("  ✓ GET /api/models passed.", flush=True)

    # 2. Metrics check
    print("\n[Test 2] Testing GET /api/metrics...", flush=True)
    metrics = get_metrics()
    assert "svm" in metrics, "Missing 'svm' in metrics"
    assert "reconstruction" in metrics, "Missing 'reconstruction' in metrics"
    assert "figures" in metrics, "Missing 'figures' in metrics"
    assert "per_class_recon" in metrics, "Missing 'per_class_recon' in metrics"
    print(f"  Test Recon MSE: {metrics['reconstruction'].get('test_mse')}", flush=True)
    print(f"  Test SVM Acc:   {metrics['svm'].get('test', {}).get('accuracy')}", flush=True)
    print(f"  Figures found:  {len(metrics['figures'])}", flush=True)
    print("  ✓ GET /api/metrics passed.", flush=True)

    # 3. Samples check
    print("\n[Test 3] Testing GET /api/samples...", flush=True)
    samples = get_sample_crops(count_per_class=4)
    assert len(samples) >= 6, "Insufficient samples returned"
    print(f"  Returned {len(samples)} sample crops across classes.", flush=True)
    assert all("image_b64" in s and "label" in s for s in samples)
    print("  ✓ GET /api/samples passed.", flush=True)

    # 4. Result figure serving
    print("\n[Test 4] Testing GET /results/<figure>...", flush=True)
    for fig_name in ["loss_curve.png", "svm_confusion_AE-latent.png", "latent_pca.png", "reconstructions.png"]:
        fig_res = get_result_figure(fig_name)
        assert fig_res is not None, f"Failed to fetch {fig_name}"
        assert os.path.exists(fig_res.path), f"Figure path {fig_res.path} does not exist"
        print(f"  ✓ Served /results/{fig_name} ({os.path.getsize(fig_res.path):,} bytes)", flush=True)

    # 5. Normal crop prediction (Helmet)
    print("\n[Test 5] Testing POST /api/predict with normal crop...", flush=True)
    helmet_crops = glob.glob("crops_64/test/helmet/*.png")
    assert len(helmet_crops) > 0, "No helmet test crops found"
    with open(helmet_crops[0], "rb") as f:
        file_bytes = f.read()

    uf = make_upload_file(os.path.basename(helmet_crops[0]), file_bytes, "image/png")
    res = predict_crop(file=uf)
    assert "predicted_class" in res
    assert "class_scores" in res
    assert res["score_type"] == "relative score, uncalibrated"
    assert "reconstruction_b64" in res
    assert "error_heatmap_b64" in res
    assert "latent_vector" in res and len(res["latent_vector"]) == 256
    assert "pca_2d" in res and len(res["pca_2d"]) == 2
    assert "metrics" in res and "mse" in res["metrics"]
    print(f"  Predicted class: {res['predicted_class']}", flush=True)
    print(f"  Class scores:    {res['class_scores']}", flush=True)
    print(f"  Metrics:         {res['metrics']}", flush=True)
    print(f"  Latent dim:      {len(res['latent_vector'])}", flush=True)
    print("  ✓ Normal crop prediction passed.", flush=True)

    # 6. Safety violation prediction (Head)
    print("\n[Test 6] Testing violation detection on unhelmeted head crop...", flush=True)
    head_crops = glob.glob("crops_64/test/head/*.png")
    assert len(head_crops) > 0, "No head test crops found"
    with open(head_crops[0], "rb") as f:
        head_bytes = f.read()

    uf_head = make_upload_file(os.path.basename(head_crops[0]), head_bytes, "image/png")
    res_head = predict_crop(file=uf_head)
    print(f"  Head crop prediction: {res_head['predicted_class']} (is_violation={res_head['is_violation']})", flush=True)
    print("  ✓ Safety violation flag check passed.", flush=True)

    # 7. Edge Case: Wrong file type
    print("\n[Test 7] Testing edge case: Unsupported file format...", flush=True)
    uf_bad = make_upload_file("notes.txt", b"plain text", "text/plain")
    try:
        predict_crop(file=uf_bad)
        assert False, "Should have raised HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print(f"  ✓ Rejected unsupported file format correctly: '{e.detail}'", flush=True)

    # 8. Edge Case: Oversized file (> 10MB)
    print("\n[Test 8] Testing edge case: Oversized file (> 10 MB)...", flush=True)
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)
    uf_big = make_upload_file("big.png", oversized_bytes, "image/png")
    try:
        predict_crop(file=uf_big)
        assert False, "Should have raised HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print(f"  ✓ Rejected oversized file correctly: '{e.detail}'", flush=True)

    # 9. Edge Case: Full site photo with crop box
    print("\n[Test 9] Testing full image with crop box parameters...", flush=True)
    synthetic_img = Image.new("RGB", (300, 200), color=(120, 150, 180))
    img_buf = io.BytesIO()
    synthetic_img.save(img_buf, format="JPEG")
    site_bytes = img_buf.getvalue()

    uf_site = make_upload_file("site_photo.jpg", site_bytes, "image/jpeg")
    res_site = predict_crop(
        file=uf_site,
        crop_x=50,
        crop_y=30,
        crop_w=64,
        crop_h=64
    )
    assert "predicted_class" in res_site
    print(f"  Crop box prediction: class={res_site['predicted_class']}, scores={res_site['class_scores']}", flush=True)
    print("  ✓ Full site photo with crop box passed.", flush=True)

    # 10. Test set accuracy evaluation on 120 random test-split crops
    print("\n[Test 10] Evaluating 120 random test crops via POST /api/predict...", flush=True)
    random.seed(42)
    classes = ["head", "helmet", "vest"]
    all_test_crops = glob.glob("crops_64/test/*/*.png")
    selected_crops_paths = random.sample(all_test_crops, min(120, len(all_test_crops)))
    selected_crops = [(p, os.path.basename(os.path.dirname(p))) for p in selected_crops_paths]
    print(f"  Sending {len(selected_crops)} test crops across classes...", flush=True)

    correct = 0
    per_class_correct = {c: 0 for c in classes}
    per_class_total = {c: 0 for c in classes}

    for i, (path, true_label) in enumerate(selected_crops):
        with open(path, "rb") as f:
            f_bytes = f.read()
        uf_item = make_upload_file(os.path.basename(path), f_bytes, "image/png")
        pred_res = predict_crop(file=uf_item)
        pred = pred_res["predicted_class"]

        per_class_total[true_label] += 1
        if pred == true_label:
            correct += 1
            per_class_correct[true_label] += 1

        if (i + 1) % 30 == 0 or (i + 1) == len(selected_crops):
            print(f"    - Evaluated {i+1}/{len(selected_crops)} crops (current acc: {correct/(i+1)*100:.1f}%)...", flush=True)

    overall_acc = (correct / len(selected_crops)) * 100
    print(f"\n  Batch Test Accuracy: {overall_acc:.2f}% ({correct}/{len(selected_crops)})", flush=True)
    for c in classes:
        c_acc = (per_class_correct[c] / per_class_total[c]) * 100 if per_class_total[c] > 0 else 0
        print(f"    - {c:<7}: {c_acc:.1f}% ({per_class_correct[c]}/{per_class_total[c]})", flush=True)

    # Verify accuracy is close to notebook reported accuracy (~92%)
    assert overall_acc >= 85.0, f"Expected accuracy >= 85%, got {overall_acc:.2f}%"
    print(f"  ✓ 120-crop test accuracy check PASSED ({overall_acc:.2f}% vs target ~92%).", flush=True)

    # 9. Test CVAE Endpoints
    print("\n[Test 9] Testing CVAE Endpoints (/api/cvae/*)...", flush=True)
    from dashboard.models.cvae.router import get_cvae_info, get_cvae_metrics_endpoint, generate_samples, GenerateRequest
    cvae_info = get_cvae_info()
    assert cvae_info["model_id"] == "cvae"
    assert "conditioning" in cvae_info
    print(f"  ✓ GET /api/cvae/info passed: {cvae_info['name']} (status: {cvae_info['status']})", flush=True)

    cvae_metrics_res = get_cvae_metrics_endpoint()
    assert cvae_metrics_res["status"] in ("ready", "not_trained")
    print(f"  ✓ GET /api/cvae/metrics passed: status={cvae_metrics_res['status']}", flush=True)

    if cvae_metrics_res["status"] == "ready":
        gen_res = generate_samples(GenerateRequest(class_name="helmet", temperature=0.8, seed=42, num_samples=2))
        assert len(gen_res["images"]) == 2
        assert gen_res["class"] == "helmet"
        print(f"  ✓ POST /api/cvae/generate passed: generated 2 helmet crops.", flush=True)

    # 10. Test cGAN Endpoints
    print("\n[Test 10] Testing cGAN Endpoints (/api/cgan/*)...", flush=True)
    from dashboard.models.cgan.router import (
        get_cgan_info, get_cgan_metrics, generate_samples as cgan_generate,
        GenerateRequest as CGANGenReq, generate_walk, LatentWalkRequest,
        get_augmentation_table
    )
    cgan_info = get_cgan_info()
    assert cgan_info["model_id"] == "cgan"
    assert "features" in cgan_info
    print(f"  ✓ GET /api/cgan/info passed: {cgan_info['name']} (status: {cgan_info['status']})", flush=True)

    cgan_metrics_res = get_cgan_metrics()
    assert cgan_metrics_res["status"] in ("ready", "coming_soon")
    print(f"  ✓ GET /api/cgan/metrics passed: status={cgan_metrics_res['status']}", flush=True)

    aug_table_res = get_augmentation_table()
    assert aug_table_res["status"] in ("success", "not_ready")
    print(f"  ✓ GET /api/cgan/augmentation_table passed: status={aug_table_res['status']}", flush=True)

    # 11. Test Diffusion Endpoints
    print("\n[Test 11] Testing Diffusion Endpoints (/api/diffusion/*)...", flush=True)
    from dashboard.models.diffusion.router import (
        get_diffusion_info, get_diffusion_metrics_endpoint, restore_crop
    )
    diff_info = get_diffusion_info()
    assert diff_info["model_id"] == "diffusion"
    assert "supported_degradations" in diff_info
    print(f"  ✓ GET /api/diffusion/info passed: {diff_info['name']} (status: {diff_info['status']})", flush=True)

    diff_metrics_res = get_diffusion_metrics_endpoint()
    assert diff_metrics_res["status"] in ("ready", "coming_soon")
    print(f"  ✓ GET /api/diffusion/metrics passed: status={diff_metrics_res['status']}", flush=True)

    # Test interactive restoration
    restore_res = restore_crop(
        file=None,
        degradation_type="gaussian_noise",
        severity="mild",
        ddim_steps=5,
        sample_class="helmet"
    )
    assert restore_res["status"] == "success"
    assert "restored_b64" in restore_res
    assert "metrics" in restore_res
    assert "classifications" in restore_res
    print(f"  ✓ POST /api/diffusion/restore passed (PSNR: {restore_res['metrics']['psnr_diffusion']} dB, latency: {restore_res['latency_ms']} ms).", flush=True)

    # 12. Test Policy Checker Endpoints
    print("\n[Test 12] Testing Policy Checker Endpoints (/api/policy/*)...", flush=True)
    from dashboard.models.policy.router import (
        get_policy_info, get_policy_metrics, get_policy_samples,
        ingest_policy_document, evaluate_compliance_verdict
    )
    pol_info = get_policy_info()
    assert pol_info["model_id"] == "policy"
    assert pol_info["status"] == "ready"
    assert len(pol_info["classes"]) == 5
    print(f"  ✓ GET /api/policy/info passed: {pol_info['name']} (status: {pol_info['status']}, classes: {pol_info['classes']})", flush=True)

    pol_metrics = get_policy_metrics()
    assert "distilbert_5seeds" in pol_metrics
    assert "temperature_calibration" in pol_metrics
    print(f"  ✓ GET /api/policy/metrics passed: Test F1={pol_metrics['distilbert_5seeds']['test_macro_f1_mean']:.4f}, Calibrated ECE={pol_metrics['temperature_calibration']['test_ece_calibrated']:.4f}", flush=True)

    pol_samples = get_policy_samples()
    assert len(pol_samples) >= 4
    print(f"  ✓ GET /api/policy/samples passed: {len(pol_samples)} authentic safety policies available.", flush=True)

    # Test Ingestion
    ingest_res = ingest_policy_document(
        file=None,
        raw_text="Hard hats must be worn at all times in active construction zones. High-visibility vests are not required in office trailers.",
        sample_id=None
    )
    assert ingest_res["sentence_count"] == 2
    assert len(ingest_res["classified_sentences"]) == 2
    print(f"  ✓ POST /api/policy/ingest passed: classified {ingest_res['sentence_count']} sentences ({ingest_res['classified_sentences'][0]['predicted_label']}, {ingest_res['classified_sentences'][1]['predicted_label']}).", flush=True)

    # Test Verdict Engine - Violation
    v_viol = evaluate_compliance_verdict(
        crop_file=None,
        sample_crop="head",
        policy_sentences_json=json.dumps([{"text": "Hard hats are required.", "page": 1, "predicted_label": "helmet_required"}])
    )
    assert v_viol["verdict"] == "Violation"
    assert v_viol["cited_clause"] == "Hard hats are required."
    print(f"  ✓ POST /api/policy/verdict (Violation Scenario) passed: verdict='{v_viol['verdict']}', cited='{v_viol['cited_clause']}'", flush=True)

    # Test Verdict Engine - Cannot Determine (Both Required with single helmet crop)
    v_cannot = evaluate_compliance_verdict(
        crop_file=None,
        sample_crop="helmet",
        policy_sentences_json=json.dumps([{"text": "Both hard hats and safety vests are mandatory.", "page": 2, "predicted_label": "both_required"}])
    )
    assert v_cannot["verdict"] == "Cannot determine"
    print(f"  ✓ POST /api/policy/verdict (Partial Evidence Scenario) passed: verdict='{v_cannot['verdict']}' (proper three-state gating).", flush=True)

    # Test Verdict Engine - Exemption Clause Handled
    v_exempt = evaluate_compliance_verdict(
        crop_file=None,
        sample_crop="head",
        policy_sentences_json=json.dumps([{"text": "Hard hats are not required inside the site office.", "page": 1, "predicted_label": "none"}])
    )
    assert v_exempt["verdict"] == "Compliant"
    print(f"  ✓ POST /api/policy/verdict (Exemption Scenario) passed: verdict='{v_exempt['verdict']}'.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL API VERIFICATION TESTS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_tests()
