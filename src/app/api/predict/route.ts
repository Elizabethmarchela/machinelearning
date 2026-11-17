import { NextResponse } from "next/server";
import { getClassifier } from "@/lib/model-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body harus berupa JSON." }, { status: 400 });
  }
  const text = typeof (body as { text?: unknown })?.text === "string" ? (body as { text: string }).text.trim() : "";
  if (!text) return NextResponse.json({ error: "Teks ulasan tidak boleh kosong." }, { status: 400 });
  if (text.length > 5000) return NextResponse.json({ error: "Teks maksimal 5000 karakter." }, { status: 413 });

  const clf = getClassifier();
  if (!clf) {
    return NextResponse.json(
      { error: "Model belum tersedia. Jalankan `python ml/train.py` terlebih dahulu." },
      { status: 503 },
    );
  }
  return NextResponse.json(clf.predict(text));
}
