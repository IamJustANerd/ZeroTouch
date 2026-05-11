# Use a lightweight Node.js environment as the base (Node 22 is required)
FROM node:22-slim

# Set up the isolated working directory inside the container
WORKDIR /app

# Install OpenClaw directly via npm to bypass the interactive shell script
RUN npm install -g openclaw@latest

# Expose OpenClaw's default local gateway port
EXPOSE 18789

# Default command to keep the container running in the background
CMD ["tail", "-f", "/dev/null"]