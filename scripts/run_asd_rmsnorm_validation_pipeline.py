"""Independent validation run; candidate and reserve isolation checked first."""
from scripts.run_asd_compile_pipeline import main
from scripts.run_asd_rmsnorm_validation_probe import WORKER_TIMEOUT


if __name__ == "__main__":
    main(study="rmsnorm_validation", folder="asd-rmsnorm-validation", worker_timeout_seconds=WORKER_TIMEOUT,
         extra_tests=("tests/test_asd_rmsnorm.py", "tests/test_asd_rmsnorm_suite.py", "tests/test_asd_rmsnorm_validation.py"))
