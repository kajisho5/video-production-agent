#!/bin/bash
# loops the exact MIX.duck engine graph and reports every distinct output
set -u
N=$1; W=$(mktemp -d); cd "$W"
ff="ffmpeg -y -hide_banner -loglevel error"
g="0.1*sin(2*PI*1000*t)*between(t\\,3\\,13)"
$ff -f lavfi -i "aevalsrc='$g|$g':s=48000:c=stereo" -t 16 -c:a pcm_s16le a.wav
$ff -f lavfi -i "aevalsrc='0.2*sin(2*PI*440*t)':s=48000:c=mono" -t 5 -c:a pcm_s16le b.wav
FULL='[0:a:0]anull[main];[1:a:0]volume=0dB,atrim=0:16.000000[music];[main]asplit=2[mainA][sc];[music]asplit=2[mdry][mwet];[sc]dcshift=shift=0.000001[key];[mwet][key]sidechaingate=threshold=0.01:ratio=9000:range=0:knee=1:attack=20:release=400[mspeech];[mspeech]volume=-0.874107[mcut];[mdry][mcut]amix=inputs=2:normalize=0[ducked];[mainA][ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mix];[mix]aformat=sample_rates=48000,apad=whole_len=768000,atrim=end_sample=768000,atrim=end=16.000010[out]'
GATE='[1:a:0]volume=0dB,atrim=0:16.000000[music];[0:a:0]dcshift=shift=0.000001[key];[music][key]sidechaingate=threshold=0.01:ratio=9000:range=0:knee=1:attack=20:release=400[out]'
SPLITKEY='[0:a:0]anull[main];[1:a:0]volume=0dB,atrim=0:16.000000[music];[main]asplit=2[mainA][sc];[sc]dcshift=shift=0.000001[key];[music][key]sidechaingate=threshold=0.01:ratio=9000:range=0:knee=1:attack=20:release=400[g];[mainA][g]amix=inputs=2:duration=first:normalize=0[out]'
win() { for s in 2.9 3.0 3.1 3.3 3.5 3.7 3.9 4.1 4.3 4.5 4.7 4.9; do printf "%s:%s " $s "$(ffmpeg -hide_banner -nostdin -nostats -ss $s -t 0.1 -i "$1" -af volumedetect -f null - 2>&1 | grep -o 'mean_volume: [-0-9.]*' | cut -d' ' -f2)"; done; echo; }
for name in FULL GATE SPLITKEY; do
  fc="${!name}"; declare -A seen=(); declare -A cnt=()
  for pre in "" "-reinit_filter 0"; do
  for i in $(seq 1 $N); do
    ffmpeg -hide_banner -loglevel verbose -nostdin -y $pre -f wav -i a.wav $pre -f wav -i b.wav -filter_complex "$fc" -map '[out]' -c:a pcm_s16le -shortest o.wav 2> err.txt
    h=$(md5sum < o.wav | cut -c1-8); k="$name[$pre] $h"; cnt[$k]=$(( ${cnt[$k]:-0} + 1 ))
    if [ -z "${seen[$k]:-}" ]; then seen[$k]=1; echo "== $k first at run $i"; win o.wav; grep -iE "reconfig|changed|discard|drop|reinit" err.txt | head -5; fi
  done; done
  for k in "${!cnt[@]}"; do echo "COUNT $k ${cnt[$k]}"; done
  unset seen cnt
done
