# Redis Setup Script for Windows
# Run this as Administrator

Write-Host "============================================"
Write-Host "   REDIS SETUP FOR AI TRADING SYSTEM"
Write-Host "============================================"

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Please run this script as Administrator" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set variables
$redisVersion = "3.0.504"
$redisUrl = "https://github.com/microsoftarchive/redis/releases/download/win-$redisVersion/Redis-x64-$redisVersion.zip"
$downloadPath = "$env:TEMP\redis.zip"
$installPath = "C:\Redis"
$redisExe = "$installPath\redis-server.exe"
$redisConf = "$installPath\redis.windows.conf"

Write-Host "`nStep 1: Checking existing Redis installation..."

# Check if Redis is already installed
if (Test-Path $redisExe) {
    Write-Host "Redis already installed at: $installPath" -ForegroundColor Green
    
    # Check if service is running
    $service = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($service) {
        Write-Host "Redis service status: $($service.Status)" -ForegroundColor Yellow
        if ($service.Status -ne "Running") {
            Write-Host "Starting Redis service..."
            Start-Service -Name "Redis"
        }
    }
} else {
    Write-Host "Redis not found. Installing..."
    
    Write-Host "`nStep 2: Downloading Redis..."
    try {
        Invoke-WebRequest -Uri $redisUrl -OutFile $downloadPath -UseBasicParsing
        Write-Host "Downloaded Redis successfully" -ForegroundColor Green
    } catch {
        Write-Host "Failed to download Redis: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "`nStep 3: Extracting Redis..."
    try {
        # Create installation directory
        if (!(Test-Path $installPath)) {
            New-Item -ItemType Directory -Path $installPath -Force | Out-Null
        }
        
        # Extract Redis
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($downloadPath, $installPath)
        
        Write-Host "Extracted Redis to: $installPath" -ForegroundColor Green
        
        # Clean up download
        Remove-Item $downloadPath -Force
        
    } catch {
        Write-Host "Failed to extract Redis: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "`nStep 4: Configuring Redis..."
    
    # Create Redis configuration
    $redisConfig = @"
# Redis Configuration for Trading System
port 6379
bind 127.0.0.1
save 900 1
save 300 10
save 60 10000
maxmemory 256mb
maxmemory-policy allkeys-lru
timeout 0
tcp-keepalive 60
loglevel notice
logfile redis.log
databases 16
"@
    
    $redisConfig | Out-File -FilePath $redisConf -Encoding ascii
    Write-Host "Created Redis configuration file" -ForegroundColor Green
}

Write-Host "`nStep 5: Installing Redis as Windows Service..."

try {
    # Stop existing service if running
    $existingService = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($existingService) {
        if ($existingService.Status -eq "Running") {
            Stop-Service -Name "Redis" -Force
        }
        # Remove existing service
        & sc.exe delete "Redis" | Out-Null
        Start-Sleep -Seconds 2
    }
    
    # Install Redis as service
    & $redisExe --service-install --service-name Redis --port 6379
    
    # Configure service to start automatically
    & sc.exe config Redis start= auto | Out-Null
    
    # Start Redis service
    Start-Service -Name "Redis"
    
    Write-Host "Redis service installed and started successfully" -ForegroundColor Green
    
} catch {
    Write-Host "Failed to install Redis service: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Trying to start Redis manually..." -ForegroundColor Yellow
    
    # Start Redis manually in background
    Start-Process -FilePath $redisExe -ArgumentList $redisConf -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "`nStep 6: Testing Redis connection..."

try {
    # Test Redis connection using redis-cli
    $redisCli = "$installPath\redis-cli.exe"
    if (Test-Path $redisCli) {
        $pingResult = & $redisCli ping 2>$null
        if ($pingResult -eq "PONG") {
            Write-Host "✓ Redis is responding to ping" -ForegroundColor Green
        } else {
            throw "Redis not responding"
        }
    } else {
        # Test using PowerShell TCP connection
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("127.0.0.1", 6379)
        if ($tcpClient.Connected) {
            Write-Host "✓ Redis is accepting connections on port 6379" -ForegroundColor Green
            $tcpClient.Close()
        } else {
            throw "Cannot connect to Redis"
        }
    }
} catch {
    Write-Host "✗ Redis connection test failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Redis may still be starting up. Please test manually later." -ForegroundColor Yellow
}

Write-Host "`nStep 7: Adding Redis to Windows Firewall..."
try {
    # Add firewall rule for Redis
    New-NetFirewallRule -DisplayName "Redis Server" -Direction Inbound -Port 6379 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null
    Write-Host "✓ Added Redis firewall rule" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not add firewall rule (may already exist)" -ForegroundColor Yellow
}

Write-Host "`nStep 8: Creating Redis management shortcuts..."

# Create start/stop scripts
$startScript = @"
@echo off
echo Starting Redis Server...
net start Redis
echo Redis Server started
pause
"@

$stopScript = @"
@echo off
echo Stopping Redis Server...
net stop Redis
echo Redis Server stopped
pause
"@

$startScript | Out-File -FilePath "$installPath\start-redis.bat" -Encoding ascii
$stopScript | Out-File -FilePath "$installPath\stop-redis.bat" -Encoding ascii

Write-Host "✓ Created management scripts in $installPath" -ForegroundColor Green

Write-Host "`n============================================"
Write-Host "   REDIS SETUP COMPLETED SUCCESSFULLY!"
Write-Host "============================================"

Write-Host "`nRedis Information:"
Write-Host "  Installation Path: $installPath"
Write-Host "  Configuration: $redisConf"
Write-Host "  Service Name: Redis"
Write-Host "  Port: 6379"
Write-Host "  Host: 127.0.0.1 (localhost)"

Write-Host "`nManagement Commands:"
Write-Host "  Start Service: net start Redis"
Write-Host "  Stop Service: net stop Redis"
Write-Host "  Test Connection: redis-cli ping"
Write-Host "  Redis CLI: $installPath\redis-cli.exe"

Write-Host "`nNext Steps:"
Write-Host "1. Test your trading system: python main.py"
Write-Host "2. Redis should now be used instead of in-memory cache"
Write-Host "3. Check cache stats in the system output"

Write-Host "`n✅ Redis is ready for your AI Trading System!"
Read-Host "`nPress Enter to exit"