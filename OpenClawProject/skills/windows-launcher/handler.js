// This runs inside the OpenClaw Docker container
async function windows_launcher_run(input) {
  const { action, path, file, app } = input;
  
  // Normalize parameters for the bridge
  const payload = {
    action: action || app,
    path: path || file
  };

  if (!payload.action) {
    return { error: "Action or App name is required." };
  }

  try {
    const response = await fetch('http://host.docker.internal:5000/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    return await response.json();
  } catch (error) {
    return { 
      error: "CONNECTION_FAILED",
      message: `Failed to connect to Windows Host: ${error.message}. The host_bridge.py server is NOT running or is blocked by a firewall. DO NOT RETRY this tool call. Tell the user they need to start host_bridge.py on their Windows machine and ensure port 5000 is open.`
    };
  }
}

// Exporting both underscore and hyphen versions just in case
module.exports = {
  windows_launcher_run,
  "windows-launcher-run": windows_launcher_run
};
