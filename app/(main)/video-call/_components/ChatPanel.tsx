"use client";
import { memo, useState } from "react";
import type React from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send } from "lucide-react";

interface Message {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: Date;
  type: "text" | "system";
}

interface ChatPanelProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
}

function ChatPanel({ messages, onSendMessage }: ChatPanelProps) {
  const [inputValue, setInputValue] = useState("");

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-slate-700/50">
        <h3 className="text-lg font-semibold">Chat</h3>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`p-3 rounded-lg ${
                message.senderId === "local"
                  ? "bg-purple-500/20 ml-4"
                  : message.type === "system"
                    ? "bg-slate-700/50 text-center text-sm"
                    : "bg-slate-700/50 mr-4"
              }`}
            >
              {message.type !== "system" && (
                <div className="text-xs text-slate-400 mb-1">
                  {message.senderName} •{" "}
                  {message.timestamp.toLocaleTimeString()}
                </div>
              )}
              <div className="text-sm">{message.content}</div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 border-t border-slate-700/50">
        <div className="flex gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            className="bg-slate-700/50 border-slate-600"
          />
          <Button onClick={handleSend} size="sm">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

export default memo(ChatPanel);
