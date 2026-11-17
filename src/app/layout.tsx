import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Analisis Sentimen Ulasan | TF-IDF + Machine Learning",
  description: "Klasifikasi sentimen ulasan berbahasa Indonesia dengan TF-IDF dan model machine learning terbaik.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id">
      <body>
        <header className="topbar">
          <div className="container bar-inner">
            <span className="brand">Sentimen Ulasan</span>
            <nav>
              <Link href="/">Prediksi</Link>
              <Link href="/perbandingan">Perbandingan Model</Link>
            </nav>
          </div>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
