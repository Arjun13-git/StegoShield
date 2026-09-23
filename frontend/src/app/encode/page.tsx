import type { Metadata } from "next";
import { EncodeWorkbench } from "@/components/encode/EncodeWorkbench";

export const metadata: Metadata = { title: "Encode" };

export default function EncodePage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Embed a message with LSB steganography</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted sm:text-base">
          A controlled demonstration of the technique the detector targets. The server embeds your text into the least-significant
          bits of a cover image and returns a PNG, which you can then send back through the detector.
        </p>
      </header>
      <EncodeWorkbench />
    </div>
  );
}
