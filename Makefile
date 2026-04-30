# Google Cloud (obligatoire pour build/push/deploy : exporter ces variables)
#   export GCP_PROJECT_ID=ton-projet-id
#   export GCP_REGION=europe-west1
#   export ARTIFACTSREPO=nom-repo-artifact-registry
#   export IMAGE=mon-image
#   export SERVICE=mon-service   # souvent = IMAGE (ex. my-api-app)
#   export MEMORY=1Gi
#
# Une fois : gcloud auth login && gcloud config set project $GCP_PROJECT_ID
#            make gcp_configure_docker
#            Artifact Registry : créer le repo s'il n'existe pas (Console GCP ou
#            gcloud artifacts repositories create $ARTIFACTSREPO --repository-format=docker --location=$GCP_REGION)
#
# Pipeline : make build_for_production push_image_production deploy_to_cloud_run
# ou en une fois : make release_cloud_run

# MODULE INSTALLS
freeze:
	pip freeze > requirements.txt

install:
	@pip install -r requirements.txt

install_package:
	@pip uninstall -y app || :
	@pip install -e .

# API
run_api:
	uvicorn app.api.fast:app --reload

build_img_local:
	docker build -t $(IMAGE):local .

gcp_configure_docker:
	@test -n "$(GCP_REGION)" && test -n "$(GCP_PROJECT_ID)" || (echo "Set GCP_REGION and GCP_PROJECT_ID"; exit 1)
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet

build_for_production:
	docker build \
		--platform linux/amd64 \
		-t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(ARTIFACTSREPO)/$(IMAGE):prod \
		.

push_image_production:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(ARTIFACTSREPO)/$(IMAGE):prod

# SERVICE = nom du service Cloud Run (1er arg obligatoire de gcloud run deploy)
deploy_to_cloud_run:
	@test -n "$(SERVICE)" && test -n "$(GCP_PROJECT_ID)" && test -n "$(MEMORY)" || (echo "Set SERVICE, GCP_PROJECT_ID, MEMORY (and other GCP_* / ARTIFACTSREPO / IMAGE)"; exit 1)
	gcloud run deploy $(SERVICE) \
		--image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(ARTIFACTSREPO)/$(IMAGE):prod \
		--project $(GCP_PROJECT_ID) \
		--region $(GCP_REGION) \
		--memory $(MEMORY) \
		--allow-unauthenticated \
		--platform managed

# Enchaînement (après gcp_configure_docker et export des variables)
release_cloud_run: build_for_production push_image_production deploy_to_cloud_run
	@echo "Déploiement terminé. URL : gcloud run services describe $(SERVICE) --region $(GCP_REGION) --format='value(status.url)'"

run_streamlit:
	streamlit run app/frontend_file.py
