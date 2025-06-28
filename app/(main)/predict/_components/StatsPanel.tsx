import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { BarChart3, Clock, Target, TrendingUp } from "lucide-react";

interface Stats {
  totalPredictions: number;
  averageConfidence: number;
  sessionDuration: number;
  topPredictions: { [key: string]: number };
}

interface StatsPanelProps {
  stats: Stats;
}

function StatsPanel({ stats }: StatsPanelProps) {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const topPredictionsArray = Object.entries(stats.topPredictions)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Predictions */}
      <Card className="bg-slate-800/50 backdrop-blur-sm border-purple-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-purple-400 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Total Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">
            {stats.totalPredictions}
          </div>
        </CardContent>
      </Card>

      {/* Average Confidence */}
      <Card className="bg-slate-800/50 backdrop-blur-sm border-cyan-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-cyan-400 flex items-center gap-2">
            <Target className="w-4 h-4" />
            Avg Confidence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">
            {stats.averageConfidence.toFixed(1)}%
          </div>
          <Progress value={stats.averageConfidence} className="mt-2 h-2" />
        </CardContent>
      </Card>

      {/* Session Duration */}
      <Card className="bg-slate-800/50 backdrop-blur-sm border-green-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-green-400 flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Session Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">
            {formatDuration(stats.sessionDuration)}
          </div>
        </CardContent>
      </Card>

      {/* Top Prediction */}
      <Card className="bg-slate-800/50 backdrop-blur-sm border-yellow-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-yellow-400 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Most Frequent
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-lg font-bold text-white truncate">
            {topPredictionsArray[0]?.[0] || "None"}
          </div>
          {topPredictionsArray[0] && (
            <div className="text-sm text-slate-400">
              {topPredictionsArray[0][1]} times
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default memo(StatsPanel);
