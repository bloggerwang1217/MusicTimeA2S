#!/bin/bash
# Download piano soundfonts for piano-base
# Usage: bash scripts/download_piano_soundfonts.sh

set -e

SOUNDFONT_DIR="data/soundfonts/piano"
mkdir -p "$SOUNDFONT_DIR"
cd "$SOUNDFONT_DIR"

echo "Downloading piano soundfonts to $SOUNDFONT_DIR..."

# TimGM6mb.sf2 (direct download)
if [ ! -f "TimGM6mb.sf2" ]; then
    echo "Downloading TimGM6mb.sf2..."
    wget -q --show-progress "https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw" -O TimGM6mb.sf2
else
    echo "TimGM6mb.sf2 already exists, skipping"
fi

# FluidR3_GM.sf2 (zip)
if [ ! -f "FluidR3_GM.sf2" ]; then
    echo "Downloading FluidR3_GM.sf2..."
    wget -q --show-progress "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip" -O FluidR3_GM.zip
    unzip -q FluidR3_GM.zip
    rm FluidR3_GM.zip
else
    echo "FluidR3_GM.sf2 already exists, skipping"
fi

# UprightPianoKW (7z)
if [ ! -f "UprightPianoKW-20220221.sf2" ]; then
    echo "Downloading UprightPianoKW-20220221.sf2..."
    wget -q --show-progress "https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-SF2-20220221.7z" -O UprightPianoKW.7z
    7z x -y UprightPianoKW.7z > /dev/null
    mv UprightPianoKW-SF2-20220221/*.sf2 . 2>/dev/null || true
    rm -rf UprightPianoKW-SF2-20220221 UprightPianoKW.7z
else
    echo "UprightPianoKW-20220221.sf2 already exists, skipping"
fi

# SalamanderGrandPiano (tar.xz)
if [ ! -f "SalamanderGrandPiano-V3+20200602.sf2" ]; then
    echo "Downloading SalamanderGrandPiano-V3+20200602.sf2..."
    wget -q --show-progress "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz" -O SalamanderGrandPiano.tar.xz
    tar -xJf SalamanderGrandPiano.tar.xz
    mv SalamanderGrandPiano-SF2-V3+20200602/*.sf2 . 2>/dev/null || true
    rm -rf SalamanderGrandPiano-SF2-V3+20200602 SalamanderGrandPiano.tar.xz
else
    echo "SalamanderGrandPiano-V3+20200602.sf2 already exists, skipping"
fi

# YDP-GrandPiano (tar.bz2)
if [ ! -f "YDP-GrandPiano-20160804.sf2" ]; then
    echo "Downloading YDP-GrandPiano-20160804.sf2..."
    wget -q --show-progress "https://freepats.zenvoid.org/Piano/YDP-GrandPiano/YDP-GrandPiano-SF2-20160804.tar.bz2" -O YDP-GrandPiano.tar.bz2
    tar -xjf YDP-GrandPiano.tar.bz2
    mv YDP-GrandPiano-SF2-20160804/*.sf2 . 2>/dev/null || true
    rm -rf YDP-GrandPiano-SF2-20160804 YDP-GrandPiano.tar.bz2
else
    echo "YDP-GrandPiano-20160804.sf2 already exists, skipping"
fi

echo ""
echo "Done! Soundfonts downloaded:"
ls -lh *.sf2
