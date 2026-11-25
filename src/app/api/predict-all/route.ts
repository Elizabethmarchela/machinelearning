
import { NextResponse } from "next/server";
import { getAllClassifiers } from "@/lib/model-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
    let body: unknown;

    try {
        body = await req.json();
    } catch {
        return NextResponse.json(
            { error: "Body harus berupa JSON." },
            { status: 400 }
        );
    }

    const data = body as {
        text?: unknown;
        selectedModel?: unknown;
    };

    const text =
        typeof data.text === "string" ? data.text.trim() : "";

    const selectedModel = data.selectedModel;

    if (!text) {
        return NextResponse.json(
            { error: "Teks ulasan tidak boleh kosong." },
            { status: 400 }
        );
    }

    if (text.length > 5000) {
        return NextResponse.json(
            { error: "Teks maksimal 5000 karakter." },
            { status: 413 }
        );
    }

    const validModels = ["mnb", "svm", "lr"];

    if (
        typeof selectedModel !== "string" ||
        !validModels.includes(selectedModel)
    ) {
        return NextResponse.json(
            { error: "Model tidak valid." },
            { status: 400 }
        );
    }

    const models = getAllClassifiers();

    if (!models) {
        return NextResponse.json(
            { error: "Model belum lengkap. Jalankan python ml/train.py terlebih dahulu." },
            { status: 503 }
        );
    }

    const selected = models.find(
        (m) => m.key === selectedModel
    );

    if (!selected) {
        return NextResponse.json(
            { error: "Model yang dipilih tidak ditemukan." },
            { status: 404 }
        );
    }

    const prediction = selected.clf.predict(text);

    return NextResponse.json({
        selectedModel: selected.key,
        prediction,
        results: models.map((m) => ({
            key: m.key,
            name: m.name,
            isBest: m.isBest,
            prediction: m.clf.predict(text)
        }))
    });
}
