# Cloudflare Tunnel Setup Guide

This guide will help you set up a Cloudflare Tunnel to make your Streamlit app publicly accessible.

## Prerequisites
- A Cloudflare account (sign up at [cloudflare.com](https://www.cloudflare.com/))
- A domain managed by Cloudflare (or use the free trycloudflare.com subdomain)
- Cloudflare Tunnel installed on your machine

## Installation

1. Install cloudflared (Cloudflare's tunneling daemon):
   ```powershell
   winget install Cloudflare.cloudflared
   ```

## Authentication

1. Log in to your Cloudflare account:
   ```powershell
   cloudflared tunnel login
   ```
   - This will open a browser window for authentication
   - Select your domain when prompted

## Create a Tunnel

1. Create a new tunnel (replace `aadhaar-insights` with your preferred name):
   ```powershell
   cloudflared tunnel create aadhaar-insights
   ```
   - Note the tunnel ID that's displayed (you'll need it for the next step)

## Configure the Tunnel

1. Create a configuration file named `cloudflared.yml` in your project root:
   ```yaml
   tunnel: aadhaar-insights
   credentials-file: C:\Users\[YourUsername]\.cloudflared\[tunnel-id].json
   ingress:
     - hostname: aadhaar-insights.yourdomain.com  # Replace with your domain
       service: http://localhost:8501
     - service: http_status:404
   ```

   - Replace `[YourUsername]` with your Windows username
   - Replace `[tunnel-id]` with the ID from the tunnel creation step
   - Replace `aadhaar-insights.yourdomain.com` with your desired subdomain

## DNS Configuration

1. Create a DNS record for your tunnel:
   ```powershell
   cloudflared tunnel route dns aadhaar-insights aadhaar-insights.yourdomain.com
   ```
   - Replace `aadhaar-insights` with your tunnel name
   - Replace `aadhaar-insights.yourdomain.com` with your subdomain

## Start the Tunnel

1. In one terminal, start your Streamlit app:
   ```powershell
   streamlit run app.py
   ```

2. In another terminal, start the Cloudflare tunnel:
   ```powershell
   cloudflared tunnel run aadhaar-insights
   ```

3. Your app will be available at: `https://aadhaar-insights.yourdomain.com`

## Running as a Windows Service (Optional)

To run the tunnel as a Windows service:

1. Create a batch file `start-tunnel.bat`:
   ```batch
   @echo off
   cd /d "%~dp0"
   cloudflared tunnel run aadhaar-insights
   ```

2. Create a Windows Scheduled Task to run this on system startup:
   - Open Task Scheduler
   - Create Basic Task
   - Set trigger to "When the computer starts"
   - Action: Start a program
   - Program/script: `C:\path\to\start-tunnel.bat`
   - Start in: `C:\path\to\your\project`

## Troubleshooting

- If you get connection errors, verify:
  - The Streamlit app is running on port 8501
  - The tunnel is properly authenticated
  - The DNS records are correctly set up
  - No firewall is blocking the connection

- To view tunnel logs:
  ```powershell
  Get-Content -Path "$env:USERPROFILE\.cloudflared\cloudflared.log" -Wait
  ```

## Security Considerations

- Keep your tunnel credentials file secure
- Use Cloudflare Access for additional security
- Set up appropriate CORS headers in your Streamlit app
- Consider rate limiting to prevent abuse
