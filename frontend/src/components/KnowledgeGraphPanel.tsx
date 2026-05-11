// Knowledge Graph Paths Panel — 文字树展示知识图谱路径
// 将图谱路径渲染为 → 分隔的链条，而非可视化图形
"use client";

interface GraphPathItem {
  path: string[];
  matched_node: string;
  matched_topics: string[];
}

interface Props {
  paths: GraphPathItem[];
  isLoading: boolean;
}

function Skeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 space-y-2 animate-pulse"
        >
          <div className="flex gap-1.5 items-center">
            <div className="h-3 bg-white/[0.06] rounded w-16" />
            <span className="text-gray-700 text-[10px]">→</span>
            <div className="h-3 bg-white/[0.06] rounded w-24" />
            <span className="text-gray-700 text-[10px]">→</span>
            <div className="h-3 bg-white/[0.06] rounded w-20" />
          </div>
          <div className="h-3 bg-white/[0.06] rounded w-1/3" />
          <div className="flex gap-1">
            <div className="h-3 bg-white/[0.06] rounded w-14" />
            <div className="h-3 bg-white/[0.06] rounded w-16" />
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center py-12">
      <div className="text-center space-y-2">
        <span className="text-3xl block opacity-30">🕸️</span>
        <p className="text-sm text-gray-500">No graph paths available</p>
        <p className="text-[11px] text-gray-700">
          Knowledge graph traversal results will appear here
        </p>
      </div>
    </div>
  );
}

export default function KnowledgeGraphPanel({ paths, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-violet-500/60 to-transparent" />
          <span className="text-xs font-medium text-violet-400 tracking-widest uppercase">
            Knowledge Graph
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-violet-500/60 to-transparent" />
        </div>
        <Skeleton />
      </div>
    );
  }

  if (paths.length === 0) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-px w-6 bg-gradient-to-r from-violet-500/60 to-transparent" />
          <span className="text-xs font-medium text-violet-400 tracking-widest uppercase">
            Knowledge Graph
          </span>
          <div className="h-px w-6 bg-gradient-to-l from-violet-500/60 to-transparent" />
        </div>
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5 flex flex-col">
      <div className="flex items-center gap-2 mb-4 shrink-0">
        <div className="h-px w-6 bg-gradient-to-r from-violet-500/60 to-transparent" />
        <span className="text-xs font-medium text-violet-400 tracking-widest uppercase">
          Knowledge Graph
        </span>
        <div className="h-px w-6 bg-gradient-to-l from-violet-500/60 to-transparent" />
        <span className="text-[10px] text-gray-500 font-mono ml-auto">
          {paths.length} path{paths.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div className="space-y-2 overflow-y-auto max-h-[500px] pr-1">
        {paths.map((item, idx) => (
          <div
            key={idx}
            className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 transition-all duration-200 hover:border-white/[0.1]"
          >
            {/* Path chain */}
            <div className="flex items-center flex-wrap gap-1 text-xs">
              {item.path.map((node, i) => (
                <span key={i} className="flex items-center gap-1">
                  <span
                    className={`px-1.5 py-0.5 rounded ${
                      node === item.matched_node
                        ? "bg-violet-500/15 text-violet-300 border border-violet-500/25 font-medium"
                        : "text-gray-400"
                    }`}
                  >
                    {node}
                  </span>
                  {i < item.path.length - 1 && (
                    <span className="text-gray-700 text-[10px]">→</span>
                  )}
                </span>
              ))}
            </div>

            {/* Matched node detail */}
            {item.matched_node && (
              <div className="mt-2 flex items-center gap-2 text-[10px] text-gray-500">
                <span className="text-violet-500/60">●</span>
                <span>
                  Matched: <span className="text-gray-400 font-medium">{item.matched_node}</span>
                </span>
              </div>
            )}

            {/* Matched topics */}
            {item.matched_topics.length > 0 && (
              <div className="mt-1.5 flex items-center gap-1.5 flex-wrap">
                <span className="text-[10px] text-gray-600">Topics:</span>
                {item.matched_topics.map((topic, ti) => (
                  <span
                    key={ti}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/15"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
