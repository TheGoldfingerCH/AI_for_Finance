from pathlib import Path


class TestBase:

    def test_requirements(self):
        """
        tests for dependency requirements
        """
        with open("requirements.txt", encoding="utf-8") as requirements_file:
            lines = requirements_file.readlines()

        installed_packages = {
            line.split("==")[0].strip().lower()
            for line in lines
            if line.strip() and not line.startswith("#")
        }

        required_packages = {
            "fastapi",
            "uvicorn",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "pytest",
        }

        missing_packages = required_packages - installed_packages
        assert not missing_packages, (
            f"Missing required packages in requirements.txt: "
            f"{sorted(missing_packages)}"
        )

    def test_env_files(self):
        """
        tests environment file keys
        """
        env_path = Path(".env")
        assert env_path.exists(), "Missing .env file"

        lines = env_path.read_text(encoding="utf-8").splitlines()
        entries = {
            line.split("=", maxsplit=1)[0].strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#") and "=" in line
        }

        required_env_keys = {"GCP_PROJECT_ID"}
        missing_keys = required_env_keys - entries
        assert not missing_keys, (
            "Missing required keys in .env: "
            f"{sorted(missing_keys)}"
        )