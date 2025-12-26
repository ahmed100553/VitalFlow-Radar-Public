#!/bin/bash
# AWR2243 Setup Quick Start Script for Linux
# Run this script to verify your system is ready for the radar driver

echo "=================================================="
echo "AWR2243 Radar Driver - Linux Setup Check"
echo "=================================================="
echo ""

# 1. Check Python version
echo "[1/7] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "  ✗ Python 3 not found. Please install Python 3.7+"
    exit 1
fi
echo "  ✓ Python found"
echo ""

# 2. Check for serial ports
echo "[2/7] Checking for USB serial devices..."
if ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    echo "  ✓ Found USB serial devices:"
    ls -l /dev/ttyUSB*
elif ls /dev/ttyACM* 1> /dev/null 2>&1; then
    echo "  ✓ Found ACM serial devices:"
    ls -l /dev/ttyACM*
else
    echo "  ⚠ No serial devices found. Please connect the AWR2243 radar."
    echo "    If already connected, check with: dmesg | grep tty"
fi
echo ""

# 3. Check dialout group membership
echo "[3/7] Checking dialout group membership..."
if groups | grep -q dialout; then
    echo "  ✓ User is in dialout group"
else
    echo "  ✗ User is NOT in dialout group"
    echo "    Run: sudo usermod -a -G dialout $USER"
    echo "    Then log out and back in"
fi
echo ""

# 4. Check pip
echo "[4/7] Checking pip..."
python3 -m pip --version
if [ $? -ne 0 ]; then
    echo "  ✗ pip not found. Install with: sudo apt install python3-pip"
    exit 1
fi
echo "  ✓ pip found"
echo ""

# 5. Check/Install dependencies
echo "[5/7] Checking Python dependencies..."
python3 -c "import serial" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ⚠ pyserial not installed"
    echo "    Installing dependencies..."
    pip install -r requirements.txt
else
    echo "  ✓ pyserial is installed"
fi
echo ""

# 6. Check permissions on serial ports
echo "[6/7] Checking serial port permissions..."
if ls /dev/ttyUSB0 1> /dev/null 2>&1; then
    if [ -r /dev/ttyUSB0 ] && [ -w /dev/ttyUSB0 ]; then
        echo "  ✓ /dev/ttyUSB0 is readable and writable"
    else
        echo "  ✗ /dev/ttyUSB0 permission denied"
        echo "    Run: sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1"
        echo "    Or add user to dialout group (see step 3)"
    fi
else
    echo "  ⚠ /dev/ttyUSB0 not found"
fi
echo ""

# 7. Display usage examples
echo "[7/7] Setup complete!"
echo ""
echo "=================================================="
echo "Usage Examples:"
echo "=================================================="
echo ""
echo "# Test with CSV simulation (no hardware needed):"
echo "  python edge_producer.py"
echo ""
echo "# Test with live radar (default ports):"
echo "  python edge_producer.py --radar"
echo ""
echo "# Test with custom ports:"
echo "  python edge_producer.py --radar --cli /dev/ttyACM0 --data /dev/ttyACM1"
echo ""
echo "# Standalone radar test:"
echo "  python example_radar_usage.py"
echo ""
echo "# Radar test with visualization:"
echo "  python example_radar_usage.py --viz"
echo ""
echo "=================================================="
echo "Troubleshooting:"
echo "=================================================="
echo ""
echo "If you encounter issues:"
echo "  1. List connected devices: ls -l /dev/tty{USB,ACM}*"
echo "  2. Check kernel messages: dmesg | tail -20"
echo "  3. Verify permissions: groups (should show 'dialout')"
echo "  4. Test serial connection: python -c 'import serial; print(serial.__version__)'"
echo ""
echo "For more help, see: docs/AWR2243_Driver_Guide.md"
echo ""
