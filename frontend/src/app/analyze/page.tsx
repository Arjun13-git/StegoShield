import type { Metadata } from "next";
import { AnalyzeWorkbench } from "@/components/analysis/AnalyzeWorkbench";

export const metadata: Metadata = { title: "Analyze" };

export default function AnalyzePage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Analyze an image for LSB steganography</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted sm:text-base">
          StegoShield scores an image with a frozen classical machine-learning detector built for pixel-domain least-significant-bit
          (LSB) embedding, and shows the statistical evidence behind the score.
        </p>
      </header>

      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-md border border-line bg-surface px-4 py-3">
          <p className="font-medium text-fg">What it looks for</p>
          <p className="mt-1 text-muted">Statistical traces of LSB replacement in pixel data, in the research setting it was tested on.</p>
        </div>
        <div className="rounded-md border border-warn/40 bg-warn/5 px-4 py-3">
          <p className="font-medium text-fg">What it cannot do</p>
          <p className="mt-1 text-muted">
            It is a weak detector in testing, does not detect JPEG-domain methods (JMiPOD, J-UNIWARD, UERD), and loses its signal
            after JPEG re-compression or noise. A low score does not prove an image is clean.
          </p>
        </div>
      </div>

      <AnalyzeWorkbench />
    </div>
  );
}
