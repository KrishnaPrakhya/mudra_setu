"use client";

import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { SettingsIcon, Palette, Volume2 } from "lucide-react";

interface Settings {
  confidenceThreshold: number;
  poseColor: string;
  leftHandColor: string;
  rightHandColor: string;
  speechRate: number;
  speechPitch: number;
  autoSpeak: boolean;
}

interface SettingsPanelProps {
  settings: Settings;
  updateSettings: (settings: Partial<Settings>) => void;
}

function SettingsPanel({ settings, updateSettings }: SettingsPanelProps) {
  return (
    <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-500/20">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <SettingsIcon className="w-5 h-5" />
          Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Confidence Threshold */}
        <div className="space-y-2">
          <Label className="text-sm font-medium text-slate-300">
            Confidence Threshold: {settings.confidenceThreshold}%
          </Label>
          <Slider
            value={[settings.confidenceThreshold]}
            onValueChange={([value]) =>
              updateSettings({ confidenceThreshold: value })
            }
            max={100}
            min={50}
            step={5}
            className="w-full"
          />
        </div>

        {/* Colors */}
        <div className="space-y-3">
          <Label className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <Palette className="w-4 h-4" />
            Landmark Colors
          </Label>
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">Pose</Label>
              <input
                type="color"
                value={settings.poseColor}
                onChange={(e) => updateSettings({ poseColor: e.target.value })}
                className="w-full h-8 rounded border border-slate-600"
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">Left Hand</Label>
              <input
                type="color"
                value={settings.leftHandColor}
                onChange={(e) =>
                  updateSettings({ leftHandColor: e.target.value })
                }
                className="w-full h-8 rounded border border-slate-600"
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">Right Hand</Label>
              <input
                type="color"
                value={settings.rightHandColor}
                onChange={(e) =>
                  updateSettings({ rightHandColor: e.target.value })
                }
                className="w-full h-8 rounded border border-slate-600"
              />
            </div>
          </div>
        </div>

        {/* Speech Settings */}
        <div className="space-y-3">
          <Label className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <Volume2 className="w-4 h-4" />
            Speech Settings
          </Label>

          <div className="space-y-2">
            <Label className="text-xs text-slate-400">
              Speech Rate: {settings.speechRate.toFixed(1)}x
            </Label>
            <Slider
              value={[settings.speechRate]}
              onValueChange={([value]) => updateSettings({ speechRate: value })}
              max={2}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-xs text-slate-400">
              Speech Pitch: {settings.speechPitch.toFixed(1)}
            </Label>
            <Slider
              value={[settings.speechPitch]}
              onValueChange={([value]) =>
                updateSettings({ speechPitch: value })
              }
              max={2}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>

          <div className="flex items-center justify-between">
            <Label className="text-xs text-slate-400">
              Auto-speak predictions
            </Label>
            <Switch
              checked={settings.autoSpeak}
              onCheckedChange={(checked) =>
                updateSettings({ autoSpeak: checked })
              }
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default memo(SettingsPanel);
