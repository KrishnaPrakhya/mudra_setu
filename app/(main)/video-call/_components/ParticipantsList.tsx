"use client";
import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { User, Mic, Video } from "lucide-react";

interface Participant {
  id: string;
  name: string;
  isLocal: boolean;
}

interface ParticipantsListProps {
  participants: Participant[];
}

function ParticipantsList({ participants }: ParticipantsListProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Participants</h3>
        <Badge variant="secondary">{participants.length}</Badge>
      </div>

      <ScrollArea className="h-64">
        <div className="space-y-2">
          {participants.map((participant) => (
            <div
              key={participant.id}
              className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <div>
                  <div className="font-medium">
                    {participant.name}
                    {participant.isLocal && (
                      <Badge variant="outline" className="ml-2 text-xs">
                        You
                      </Badge>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <div className="w-4 h-4 text-green-500">
                  <Mic className="w-full h-full" />
                </div>
                <div className="w-4 h-4 text-green-500">
                  <Video className="w-full h-full" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      {participants.length === 1 && (
        <div className="text-center text-sm text-slate-400 py-4">
          Share the room ID to invite others to join
        </div>
      )}
    </div>
  );
}

export default memo(ParticipantsList);
