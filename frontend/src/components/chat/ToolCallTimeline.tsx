"use client";

import type { CopilotToolEvent } from "@/types";

interface ToolCallTimelineProps {
  events: CopilotToolEvent[];
}

export default function ToolCallTimeline({ events }: ToolCallTimelineProps) {
  if (!events.length) return null;

  return (
    <div className="space-y-1 border-t border-slate-300/15 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">Tool Activity</p>
      {events.slice(-6).map((event, index) => (
        <div key={`${event.tool}-${index}`} className="flex items-center justify-between text-[11px]">
          <span className="text-slate-300">{event.tool}</span>
          <span
            className={`rounded-full px-2 py-0.5 ${
              event.status === "running"
                ? "bg-cyan-500/15 text-cyan-200"
                : event.status === "done"
                  ? "bg-emerald-500/15 text-emerald-200"
                  : "bg-rose-500/15 text-rose-200"
            }`}
          >
            {event.status}
          </span>
        </div>
      ))}
    </div>
  );
}
