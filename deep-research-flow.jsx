import { useState } from "react";

const COLORS = {
  bg: "#08080D",
  surface: "#111118",
  surfaceHover: "#1A1A24",
  surfaceActive: "#1E1E2A",
  border: "#232334",
  borderLight: "#2E2E42",
  text: "#E2E2EC",
  textSecondary: "#9898B0",
  textDim: "#5C5C74",
  accent: "#7B6CF0",
  accentSoft: "rgba(123,108,240,0.12)",
  accentGlow: "rgba(123,108,240,0.25)",
  green: "#34D399",
  greenSoft: "rgba(52,211,153,0.10)",
  orange: "#FB923C",
  orangeSoft: "rgba(251,146,60,0.10)",
  red: "#F87171",
  redSoft: "rgba(248,113,113,0.10)",
  blue: "#60A5FA",
  blueSoft: "rgba(96,165,250,0.10)",
  cyan: "#22D3EE",
  cyanSoft: "rgba(34,211,238,0.10)",
};

const STATUS = {
  searching: { color: COLORS.blue, bg: COLORS.blueSoft, label: "搜索中", icon: "◌" },
  reading: { color: COLORS.orange, bg: COLORS.orangeSoft, label: "阅读中", icon: "◑" },
  analyzing: { color: COLORS.accent, bg: COLORS.accentSoft, label: "分析中", icon: "◈" },
  complete: { color: COLORS.green, bg: COLORS.greenSoft, label: "已完成", icon: "✓" },
  error: { color: COLORS.red, bg: COLORS.redSoft, label: "出错", icon: "✕" },
};

const TYPE_META = {
  root: { icon: "◆", label: "研究主题" },
  search: { icon: "⌕", label: "搜索任务" },
  read: { icon: "▤", label: "阅读分析" },
  analysis: { icon: "◇", label: "综合分析" },
};

const researchData = {
  id: "root",
  title: "AI Agent 在企业级应用中的最新发展趋势",
  status: "analyzing",
  type: "root",
  summary: "研究 AI Agent 在 2024-2025 年的技术演进、商业落地和行业影响",
  detail: "本次研究聚焦于 AI Agent 技术的三个核心维度：底层技术架构的快速演进（多智能体协作、工具调用标准化、长期记忆）、在不同垂直行业的商业化落地进展，以及由此引发的安全与治理挑战。\n\n研究将综合学术论文、行业报告、产品文档和新闻报道等多类信息源，最终形成一份结构化的趋势分析报告。",
  sources: [],
  findings: [
    "多 Agent 框架正在从实验阶段走向生产环境",
    "MCP 协议有望成为工具调用的事实标准",
    "安全沙箱和权限管理成为部署的核心关注点",
  ],
  children: [
    {
      id: "n1",
      title: "技术架构演进",
      status: "complete",
      type: "search",
      summary: "多 Agent 协作框架、工具调用与记忆机制",
      detail: "通过搜索学术论文和开源项目，梳理 AI Agent 底层技术栈的最新进展。重点关注三个方向：多智能体协作框架的设计范式、工具调用（Tool Use）的标准化进程、以及长期记忆与上下文管理的技术方案。",
      sources: [
        { url: "arxiv.org/abs/2402.xxxxx", title: "A Survey of Multi-Agent LLM Systems", type: "论文" },
        { url: "github.com/microsoft/autogen", title: "AutoGen - Multi-Agent Framework", type: "代码库" },
        { url: "huggingface.co/blog/agents", title: "Building AI Agents with HuggingFace", type: "博客" },
      ],
      findings: [
        "AutoGen、CrewAI、LangGraph 形成三足鼎立格局",
        "从静态 DAG 到动态图的演进趋势明显",
        "Agent 间通信协议尚未统一",
      ],
      children: [
        {
          id: "n1-1",
          title: "多 Agent 协作框架对比",
          status: "complete",
          type: "analysis",
          summary: "AutoGen vs CrewAI vs LangGraph 架构差异",
          detail: "对三大主流多 Agent 框架进行了深度对比分析。AutoGen 采用对话驱动的 Agent 协作模式，适合开放式任务；CrewAI 以角色和任务为核心抽象，更适合结构化工作流；LangGraph 基于状态图模型，提供最细粒度的流程控制。\n\n在生产环境中，LangGraph 因其可观测性和确定性执行路径，正获得更多企业用户的青睐。",
          sources: [
            { url: "blog.langchain.dev/langgraph", title: "LangGraph: 构建有状态的 AI 应用", type: "官方文档" },
            { url: "docs.crewai.com", title: "CrewAI Documentation", type: "官方文档" },
          ],
          findings: [
            "LangGraph 在企业场景中采用率最高",
            "CrewAI 上手最简单但定制性不足",
            "AutoGen 在研究场景中表现突出",
          ],
          children: [],
        },
        {
          id: "n1-2",
          title: "工具调用标准化",
          status: "complete",
          type: "read",
          summary: "MCP 协议与 Function Calling 规范",
          detail: "阅读并分析了 Anthropic 发布的 Model Context Protocol (MCP) 规范，以及 OpenAI Function Calling 的最新更新。MCP 试图定义一套通用的 Agent-工具交互标准，使得不同 LLM 提供商的 Agent 能够复用同一套工具生态。",
          sources: [
            { url: "modelcontextprotocol.io", title: "MCP 官方规范", type: "规范文档" },
            { url: "platform.openai.com/docs/function-calling", title: "OpenAI Function Calling", type: "API 文档" },
          ],
          findings: [
            "MCP 已获得多家厂商支持",
            "工具描述的标准化是关键瓶颈",
            "安全认证机制仍在制定中",
          ],
          children: [
            {
              id: "n1-2-1",
              title: "MCP 生态系统分析",
              status: "complete",
              type: "read",
              summary: "社区采用情况与生态发展",
              detail: "通过 GitHub 数据和社区反馈，评估 MCP 协议在实际开发中的采用率和生态建设情况。目前已有超过 200 个开源 MCP server 实现，覆盖数据库、API、文件系统等常见工具类型。",
              sources: [
                { url: "github.com/topics/mcp-server", title: "GitHub MCP Server Topic", type: "代码库" },
              ],
              findings: [
                "社区活跃度持续增长",
                "企业级 MCP server 质量参差不齐",
              ],
              children: [],
            },
          ],
        },
        {
          id: "n1-3",
          title: "长期记忆与上下文管理",
          status: "reading",
          type: "read",
          summary: "RAG 与向量数据库最新进展",
          detail: "正在阅读关于 Agent 长期记忆系统的最新研究。重点关注：基于 RAG 的外部记忆方案、向量数据库的性能比较（Pinecone vs Weaviate vs Qdrant），以及新兴的混合记忆架构。",
          sources: [
            { url: "arxiv.org/abs/2404.xxxxx", title: "Memory-Augmented LLM Agents", type: "论文" },
          ],
          findings: [],
          children: [],
        },
      ],
    },
    {
      id: "n2",
      title: "商业落地案例",
      status: "searching",
      type: "search",
      summary: "客服、编程、数据分析领域部署案例",
      detail: "搜索 AI Agent 在主要垂直领域的商业化部署案例，重点关注编程助手和企业自动化两个方向。目标是梳理出当前最成熟的落地场景和尚未被充分开发的机会领域。",
      sources: [
        { url: "techcrunch.com", title: "TechCrunch AI Coverage", type: "新闻" },
        { url: "bloomberg.com/technology", title: "Bloomberg Tech", type: "新闻" },
      ],
      findings: [
        "编程助手已成为最成熟的 Agent 应用",
        "企业客服是增长最快的细分市场",
      ],
      children: [
        {
          id: "n2-1",
          title: "编程助手类 Agent",
          status: "searching",
          type: "search",
          summary: "Cursor, Copilot Workspace, Devin 对比",
          detail: "正在搜索和收集主流编程 Agent 产品的最新信息，包括功能更新、用户反馈和市场数据。",
          sources: [],
          findings: [],
          children: [],
        },
        {
          id: "n2-2",
          title: "企业自动化 Agent",
          status: "searching",
          type: "search",
          summary: "Salesforce Einstein, ServiceNow 等企业级产品",
          detail: "正在搜索大型企业软件公司的 AI Agent 产品布局。",
          sources: [],
          findings: [],
          children: [],
        },
      ],
    },
    {
      id: "n3",
      title: "安全与治理挑战",
      status: "analyzing",
      type: "analysis",
      summary: "安全风险、对齐问题与监管动态",
      detail: "综合分析 AI Agent 自主行动引发的安全挑战。包括技术层面的权限控制与沙箱隔离，以及政策层面的监管框架和合规要求。",
      sources: [
        { url: "nist.gov/ai", title: "NIST AI Risk Management Framework", type: "政策文档" },
      ],
      findings: [
        "沙箱隔离是目前最被广泛采用的安全方案",
        "EU AI Act 对 Agent 的监管框架尚不明确",
      ],
      children: [
        {
          id: "n3-1",
          title: "Agent 安全红线与沙箱机制",
          status: "reading",
          type: "read",
          summary: "权限控制与安全边界设计模式",
          detail: "正在阅读关于 AI Agent 安全运行时环境的技术文档和研究论文。重点关注容器化沙箱、权限最小化原则、操作审计日志等安全模式。",
          sources: [
            { url: "arxiv.org/abs/2403.xxxxx", title: "Sandboxing LLM Agents", type: "论文" },
          ],
          findings: [],
          children: [],
        },
      ],
    },
  ],
};

// ── Helpers ──
function findNode(tree, id) {
  if (tree.id === id) return tree;
  for (const child of tree.children || []) {
    const found = findNode(child, id);
    if (found) return found;
  }
  return null;
}

function countAll(node) {
  let total = 0, done = 0;
  (function walk(n) { total++; if (n.status === "complete") done++; (n.children || []).forEach(walk); })(node);
  return { total, done };
}

function countDescendants(node) {
  if (!node.children?.length) return 0;
  return node.children.reduce((s, c) => s + 1 + countDescendants(c), 0);
}

// ── Sidebar Tree Node ──
function TreeNode({ node, depth, selectedId, onSelect, collapsed, toggleCollapse }) {
  const st = STATUS[node.status];
  const isSelected = node.id === selectedId;
  const hasChildren = node.children?.length > 0;
  const isCollapsed = collapsed[node.id];
  const isActive = node.status !== "complete" && node.status !== "error";

  return (
    <>
      <div
        onClick={() => onSelect(node.id)}
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 6,
          padding: "7px 10px 7px " + (14 + depth * 16) + "px",
          cursor: "pointer",
          background: isSelected ? COLORS.surfaceActive : "transparent",
          borderRight: isSelected ? `2px solid ${st.color}` : "2px solid transparent",
          transition: "all 0.15s ease",
          position: "relative",
        }}
        onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.background = COLORS.surfaceHover; }}
        onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.background = "transparent"; }}
      >
        {hasChildren ? (
          <button
            onClick={(e) => { e.stopPropagation(); toggleCollapse(node.id); }}
            style={{
              width: 14, height: 14, border: "none", background: "transparent",
              color: COLORS.textDim, fontSize: 8, cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              marginTop: 3, flexShrink: 0,
              transform: isCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
              transition: "transform 0.15s",
            }}
          >▾</button>
        ) : (
          <div style={{ width: 14, flexShrink: 0 }} />
        )}

        <div style={{ marginTop: 5, flexShrink: 0, position: "relative" }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%", background: st.color,
            ...(isActive ? { boxShadow: `0 0 6px ${st.color}80` } : {}),
          }} />
          {isActive && (
            <div style={{
              position: "absolute", inset: -2,
              borderRadius: "50%", border: `1px solid ${st.color}40`,
              animation: "ripple 2s ease-out infinite",
            }} />
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 12.5, fontWeight: isSelected ? 550 : 420,
            color: isSelected ? COLORS.text : COLORS.textSecondary,
            lineHeight: 1.35,
            overflow: "hidden", textOverflow: "ellipsis",
            display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
          }}>
            {node.title}
          </div>
          <div style={{
            fontSize: 11, color: COLORS.textDim, marginTop: 2,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {node.summary}
          </div>
          {isCollapsed && hasChildren && (
            <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 3 }}>
              {countDescendants(node)} 个子节点
            </div>
          )}
        </div>
      </div>

      {hasChildren && !isCollapsed && (
        <div style={{ position: "relative" }}>
          <div style={{
            position: "absolute", left: 21 + depth * 16, top: 0, bottom: 0, width: 1,
            background: `linear-gradient(to bottom, ${COLORS.border}, transparent)`,
          }} />
          {node.children.map((c) => (
            <TreeNode key={c.id} node={c} depth={depth + 1}
              selectedId={selectedId} onSelect={onSelect}
              collapsed={collapsed} toggleCollapse={toggleCollapse}
            />
          ))}
        </div>
      )}
    </>
  );
}

// ── Section Header ──
function SectionHeader({ label }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 600, color: COLORS.textDim,
      textTransform: "uppercase", letterSpacing: "0.06em",
      marginBottom: 10, paddingBottom: 6,
      borderBottom: `1px solid ${COLORS.border}`,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {label}
    </div>
  );
}

// ── Detail Panel ──
function DetailPanel({ node, onNavigate }) {
  if (!node) {
    return (
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        color: COLORS.textDim, fontSize: 13, flexDirection: "column", gap: 8,
      }}>
        <span style={{ fontSize: 28, opacity: 0.3 }}>◇</span>
        <span>选择左侧节点查看详情</span>
      </div>
    );
  }

  const st = STATUS[node.status];
  const tm = TYPE_META[node.type] || TYPE_META.search;

  return (
    <div key={node.id} style={{
      flex: 1, overflowY: "auto", padding: "28px 36px 60px",
      animation: "fadeSlide 0.25s ease",
    }}>
      {/* Type + Status badges */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <span style={{
          fontSize: 11, color: COLORS.textDim,
          background: COLORS.surface, border: `1px solid ${COLORS.border}`,
          padding: "3px 10px", borderRadius: 20,
          display: "inline-flex", alignItems: "center", gap: 4,
        }}>
          <span style={{ fontSize: 10 }}>{tm.icon}</span> {tm.label}
        </span>
        <span style={{
          fontSize: 11, color: st.color, background: st.bg,
          padding: "3px 10px", borderRadius: 20,
          display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 500,
        }}>
          <span style={{
            fontSize: 9, display: "inline-block",
            ...(node.status !== "complete" && node.status !== "error"
              ? { animation: "spin 2s linear infinite" } : {}),
          }}>{st.icon}</span>
          {st.label}
        </span>
      </div>

      {/* Title */}
      <h1 style={{
        fontSize: 22, fontWeight: 650, color: COLORS.text, margin: "0 0 8px",
        lineHeight: 1.35, letterSpacing: "-0.01em",
      }}>
        {node.title}
      </h1>

      {/* Summary */}
      <p style={{
        fontSize: 14, color: COLORS.textSecondary, lineHeight: 1.65, margin: "0 0 28px",
      }}>
        {node.summary}
      </p>

      {/* Detail */}
      {node.detail && (
        <section style={{ marginBottom: 28 }}>
          <SectionHeader label="详细说明" />
          <div style={{
            fontSize: 13.5, color: COLORS.textSecondary, lineHeight: 1.75,
            whiteSpace: "pre-wrap",
          }}>
            {node.detail}
          </div>
        </section>
      )}

      {/* Sources */}
      {node.sources?.length > 0 && (
        <section style={{ marginBottom: 28 }}>
          <SectionHeader label={`信息来源 (${node.sources.length})`} />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {node.sources.map((s, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "10px 14px", borderRadius: 8,
                background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                transition: "border-color 0.15s",
              }}
                onMouseEnter={(e) => e.currentTarget.style.borderColor = COLORS.borderLight}
                onMouseLeave={(e) => e.currentTarget.style.borderColor = COLORS.border}
              >
                <div style={{
                  width: 28, height: 28, borderRadius: 6,
                  background: COLORS.cyanSoft, color: COLORS.cyan,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, flexShrink: 0,
                }}>⊞</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 12.5, color: COLORS.text, fontWeight: 480,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {typeof s === "string" ? s : s.title}
                  </div>
                  {typeof s === "object" && s.url && (
                    <div style={{
                      fontSize: 11, color: COLORS.textDim, marginTop: 1,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>{s.url}</div>
                  )}
                </div>
                {typeof s === "object" && s.type && (
                  <span style={{
                    fontSize: 10, color: COLORS.textDim, padding: "2px 8px",
                    borderRadius: 10, background: COLORS.bg, flexShrink: 0,
                  }}>{s.type}</span>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Findings */}
      {node.findings?.length > 0 && (
        <section style={{ marginBottom: 28 }}>
          <SectionHeader label={`关键发现 (${node.findings.length})`} />
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {node.findings.map((f, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "flex-start", gap: 10,
                padding: "9px 14px", borderRadius: 8,
                background: COLORS.greenSoft, border: `1px solid ${COLORS.green}15`,
              }}>
                <span style={{
                  color: COLORS.green, fontSize: 11, marginTop: 1, flexShrink: 0, fontWeight: 600,
                }}>→</span>
                <span style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.5 }}>{f}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Children as clickable cards */}
      {node.children?.length > 0 && (
        <section>
          <SectionHeader label={`子研究节点 (${node.children.length})`} />
          <div style={{ display: "grid", gap: 6 }}>
            {node.children.map((c) => {
              const cst = STATUS[c.status];
              return (
                <div key={c.id}
                  onClick={() => onNavigate(c.id)}
                  style={{
                    padding: "12px 14px", borderRadius: 8,
                    background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                    display: "flex", alignItems: "center", gap: 10,
                    cursor: "pointer", transition: "all 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = COLORS.borderLight;
                    e.currentTarget.style.background = COLORS.surfaceHover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = COLORS.border;
                    e.currentTarget.style.background = COLORS.surface;
                  }}
                >
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: cst.color, flexShrink: 0,
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: 12.5, color: COLORS.text, fontWeight: 480,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{c.title}</div>
                    <div style={{
                      fontSize: 11, color: COLORS.textDim, marginTop: 2,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{c.summary}</div>
                  </div>
                  <span style={{
                    fontSize: 10, color: cst.color, background: cst.bg,
                    padding: "2px 8px", borderRadius: 10, flexShrink: 0,
                  }}>{cst.label}</span>
                  <span style={{ color: COLORS.textDim, fontSize: 12 }}>›</span>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

// ── Main ──
export default function DeepResearchFlow() {
  const [selectedId, setSelectedId] = useState("root");
  const [collapsed, setCollapsed] = useState({});

  const toggleCollapse = (id) => setCollapsed((p) => ({ ...p, [id]: !p[id] }));
  const selectedNode = findNode(researchData, selectedId);
  const { total, done } = countAll(researchData);
  const pct = Math.round((done / total) * 100);

  return (
    <div style={{
      display: "flex", height: "100vh", width: "100%",
      background: COLORS.bg, color: COLORS.text,
      fontFamily: "'Noto Sans SC', -apple-system, sans-serif",
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes ripple { 0% { transform: scale(1); opacity: 0.6; } 100% { transform: scale(2.2); opacity: 0; } }
        @keyframes fadeSlide { from { opacity: 0; transform: translateX(8px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        * { box-sizing: border-box; margin: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }
      `}</style>

      {/* Left Sidebar */}
      <div style={{
        width: 320, minWidth: 280, flexShrink: 0,
        background: COLORS.surface,
        borderRight: `1px solid ${COLORS.border}`,
        display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        {/* Header */}
        <div style={{ padding: "18px 16px 14px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: COLORS.accent,
              boxShadow: `0 0 10px ${COLORS.accentGlow}`,
              animation: "pulse 2.5s ease-in-out infinite",
            }} />
            <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.text }}>
              Deep Research
            </span>
            <span style={{
              fontSize: 10, color: COLORS.textDim,
              fontFamily: "'JetBrains Mono', monospace",
            }}>进行中</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              flex: 1, height: 3, background: COLORS.border, borderRadius: 2, overflow: "hidden",
            }}>
              <div style={{
                width: `${pct}%`, height: "100%", borderRadius: 2,
                background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.green})`,
                transition: "width 0.5s ease",
              }} />
            </div>
            <span style={{
              fontSize: 10, color: COLORS.textDim,
              fontFamily: "'JetBrains Mono', monospace",
            }}>{done}/{total}</span>
          </div>
        </div>

        <div style={{ height: 1, background: COLORS.border }} />

        {/* Tree */}
        <div style={{ flex: 1, overflowY: "auto", paddingTop: 6, paddingBottom: 20 }}>
          <TreeNode
            node={researchData} depth={0}
            selectedId={selectedId} onSelect={setSelectedId}
            collapsed={collapsed} toggleCollapse={toggleCollapse}
          />
        </div>
      </div>

      {/* Right Detail */}
      <DetailPanel node={selectedNode} onNavigate={setSelectedId} />
    </div>
  );
}
