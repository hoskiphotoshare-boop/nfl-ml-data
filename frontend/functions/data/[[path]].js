// Same-origin data proxy.
// Clients request /data/<path>; this function fetches the corresponding
// repository file server-side so no upstream URLs are exposed in the app.

export async function onRequest(context) {
  const { params } = context;
  const parts = Array.isArray(params.path) ? params.path : [params.path];
  const path = parts.filter(Boolean).join("/");
  if (!path) {
    return new Response("Not found", { status: 404 });
  }

  const upstream = `https://raw.githubusercontent.com/hoskiphotoshare-boop/nfl-ml-data/main/${path}`;
  const res = await fetch(upstream, {
    cf: { cacheTtl: 30, cacheEverything: true },
  });

  if (!res.ok) {
    return new Response("Not found", { status: res.status });
  }

  const body = await res.arrayBuffer();
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": res.headers.get("content-type") || "application/json",
      "Cache-Control": "public, max-age=30, must-revalidate",
      "Access-Control-Allow-Origin": "*",
    },
  });
}