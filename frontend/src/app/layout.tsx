import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HK-FinReg AI — Multi-Agent Compliance Engine",
  description: "Hong Kong FinTech Regulatory AI Platform powered by LangGraph Multi-Agent System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
