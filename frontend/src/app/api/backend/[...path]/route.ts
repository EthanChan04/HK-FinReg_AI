import { NextRequest, NextResponse } from "next/server";

const BACKEND_API_BASE =
  process.env.BACKEND_API_BASE || "http://127.0.0.1:8000";
const BACKEND_API_KEY = process.env.BACKEND_API_KEY || "";

const ALLOWED_PATTERNS: RegExp[] = [
  /^\/api\/v1\/bank-account\/verify\/stream$/,
  /^\/api\/v1\/cross-border\/assess\/stream$/,
  /^\/api\/v1\/research\/analyze$/,
  /^\/api\/v1\/review-queue\/pending$/,
  /^\/api\/v1\/review-queue\/[^/]+\/resume$/,
  /^\/api\/v1\/review-queue\/[^/]+\/reject$/,
  /^\/api\/v1\/kag\/obligation-map$/,
  /^\/api\/v1\/kag\/graph\/search$/,
  /^\/api\/v1\/copilot\/chat\/stream$/,
];

function isPathAllowed(pathname: string): boolean {
  return ALLOWED_PATTERNS.some((pattern) => pattern.test(pathname));
}

async function proxy(request: NextRequest, pathSegments: string[]): Promise<Response> {
  const upstreamPath = `/${pathSegments.join("/")}`;
  if (!isPathAllowed(upstreamPath)) {
    return NextResponse.json({ detail: "Path is not allowed by proxy policy." }, { status: 403 });
  }

  const upstreamUrl = new URL(`${BACKEND_API_BASE}${upstreamPath}`);
  upstreamUrl.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  const accept = request.headers.get("accept");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (accept) {
    headers.set("accept", accept);
  }
  if (BACKEND_API_KEY) {
    headers.set("authorization", `Bearer ${BACKEND_API_KEY}`);
  }

  let body: BodyInit | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    const rawBody = await request.arrayBuffer();
    if (rawBody.byteLength > 0) {
      body = rawBody;
    }
  }

  const init: RequestInit = {
    method: request.method,
    headers,
    cache: "no-store",
    redirect: "manual",
    body,
  };

  const upstreamResponse = await fetch(upstreamUrl.toString(), init);

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: upstreamResponse.headers,
  });
}

type Context = { params: Promise<{ path: string[] }> };

async function handle(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  if (!Array.isArray(path) || path.length === 0) {
    return NextResponse.json({ detail: "Missing upstream path." }, { status: 400 });
  }
  return proxy(request, path);
}

export async function GET(request: NextRequest, context: Context): Promise<Response> {
  return handle(request, context);
}

export async function POST(request: NextRequest, context: Context): Promise<Response> {
  return handle(request, context);
}
