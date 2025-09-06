# financial-agent-demo


To perform a multi-platform build fo the agent and push to docker hub.
```bash
# Logout and re-login to make sure creds are fresh
podman logout docker.io
podman login docker.io

# Build frontend for multiple platforms
cd frontend
podman build --platform linux/amd64,linux/arm64 --manifest samuelgrummons/gagent-frontend:latest .
podman manifest push samuelgrummons/gagent-frontend:latest docker://docker.io/samuelgrummons/gagent-frontend:latest
cd ..

# Build backend for multiple platforms  
cd backend
podman build --platform linux/amd64,linux/arm64 --manifest samuelgrummons/gagent-backend:latest .
podman manifest push samuelgrummons/gagent-backend:latest docker://docker.io/samuelgrummons/gagent-backend:latest
cd ..
```