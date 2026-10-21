"""Bounded two-GPU full development RMSNorm comparison."""
from scripts.run_asd_compile_pipeline import main
from scripts.run_asd_rmsnorm_suite_probe import WORKER_TIMEOUT


if __name__ == "__main__":
    main(study="rmsnorm_suite", folder="asd-rmsnorm-suite", worker_timeout_seconds=WORKER_TIMEOUT,
         extra_tests=("tests/test_asd_rmsnorm.py", "tests/test_asd_rmsnorm_suite.py"))
