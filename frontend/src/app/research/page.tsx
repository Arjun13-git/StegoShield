import type { Metadata } from "next";
import { Card, CardHeading } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ConfusionMatrix } from "@/components/research/ConfusionMatrix";
import { DataTable, Td, Th } from "@/components/research/DataTable";
import { PayloadChart } from "@/components/research/PayloadChart";
import { ResearchModelPanel } from "@/components/research/ResearchModelPanel";
import { CANDIDATE_LABEL, EXTERNAL_LABEL, ROBUSTNESS_LABEL, fixed, interval, research, signed } from "@/lib/research";
import { percent } from "@/lib/utils";

export const metadata: Metadata = { title: "Research" };

const p1a = research.phase1a;
const p1b = research.phase1b;
const totalSources = Object.values(p1a.sources_per_split).reduce((a, b) => a + b, 0);
const rf = p1a.candidates.RandomForest.validation;

export default function ResearchPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Research context</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted sm:text-base">
          What the detector was built on, how it was evaluated, and where it fails. Every number on this page was copied from the
          project&rsquo;s generated experiment reports; nothing is estimated or illustrative.
        </p>
        <p className="mt-2 text-xs text-muted">
          Source reports: {research.generated_from.join(", ")} (regenerate with <code className="font-mono">scripts/export_research_summary.py</code>).
        </p>
      </header>

      <Card>
        <CardHeading eyebrow="Headline" title="A weak, narrow detector" />
        <p className="text-sm leading-relaxed text-fg/90">
          On held-out BOSSBase images the deployed Random Forest reaches ROC-AUC {fixed(p1a.test.roc_auc)} at 0.10 bits per pixel
          (chance is 0.5). It has a small signal for stronger payloads, no measurable ability on ALASKA2&rsquo;s native JPEG-domain
          methods, and it loses its signal after JPEG re-compression or noise.
        </p>
      </Card>

      <section aria-labelledby="dataset" className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeading eyebrow="Phase 1A" title="Dataset" />
          <h2 id="dataset" className="sr-only">Dataset and method</h2>
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-fg/90">
            <li>BOSSBase 1.01 grayscale 512×512 images; exact duplicates excluded, leaving {totalSources.toLocaleString("en-US")} unique source images.</li>
            <li>
              Split by <strong>source image</strong> (train / validation / test = {p1a.sources_per_split.train.toLocaleString("en-US")} /{" "}
              {p1a.sources_per_split.val.toLocaleString("en-US")} / {p1a.sources_per_split.test.toLocaleString("en-US")}): a cover and its
              stego versions always share a split, so no image identity leaks across partitions.
            </li>
            <li>Stego images are made by controlled LSB replacement at set payloads (bits per pixel), generated in memory from a fixed seed ({p1a.seed}).</li>
          </ul>
        </Card>
        <Card>
          <CardHeading eyebrow="Method" title="Detection approach" />
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-fg/90">
            <li>
              {research.model.n_features} handcrafted pixel and bit-plane statistics (feature schema {research.model.feature_schema_version}): LSB
              ratio and transitions, local LSB variance, intensity mean, spread and entropy, neighbour differences and high-pass residuals.
            </li>
            <li>
              Classical models only (no deep learning): Logistic Regression, SVM and Random Forest were compared on the same features,
              and the model was chosen by validation F1, not by test results.
            </li>
            <li>The test split was evaluated once, for the chosen model.</li>
          </ul>
        </Card>
      </section>

      <Card>
        <CardHeading eyebrow="Phase 1A" title="Model comparison" />
        <p className="mb-3 text-sm text-muted">
          Validation-split metrics at 0.10 bits per pixel ({p1a["samples_per_split_at_0.10_bpp"].val.toLocaleString("en-US")} images). Selection rule: {p1a.selection_rule}.
        </p>
        <DataTable
          caption="Validation metrics for each candidate model"
          head={
            <>
              <Th>Model</Th>
              <Th>Accuracy</Th>
              <Th>Precision</Th>
              <Th>Recall</Th>
              <Th>F1</Th>
              <Th>ROC-AUC</Th>
            </>
          }
        >
          {Object.entries(p1a.candidates).map(([name, c]) => (
            <tr key={name}>
              <Th scope="row" className="font-medium text-fg">
                {CANDIDATE_LABEL[name] ?? name}{" "}
                {name === p1a.selected_model && <Badge tone="accent" className="ml-1">Selected</Badge>}
              </Th>
              <Td>{fixed(c.validation.accuracy)}</Td>
              <Td>{fixed(c.validation.precision)}</Td>
              <Td>{fixed(c.validation.recall)}</Td>
              <Td>{fixed(c.validation.f1)}</Td>
              <Td>{fixed(c.validation.roc_auc)}</Td>
            </tr>
          ))}
        </DataTable>
        <p className="mt-2 text-xs text-muted">Random Forest validation ROC-AUC {fixed(rf.roc_auc)}; the differences between candidates are small.</p>
      </Card>

      <Card>
        <CardHeading eyebrow="Phase 1A" title="Held-out test result (Random Forest, 0.10 bpp)" />
        <div className="grid gap-6 md:grid-cols-[auto_1fr] md:items-center">
          <ConfusionMatrix
            matrix={p1a.test.confusion_matrix}
            caption={`Test split, ${p1a.test.n_samples.toLocaleString("en-US")} images (half cover, half stego)`}
          />
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {(
              [
                ["Accuracy", p1a.test.accuracy],
                ["Precision", p1a.test.precision],
                ["Recall", p1a.test.recall],
                ["F1", p1a.test.f1],
                ["ROC-AUC", p1a.test.roc_auc],
              ] as const
            ).map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs text-muted">{k}</dt>
                <dd className="font-mono text-lg tabular-nums text-fg">{fixed(v)}</dd>
              </div>
            ))}
          </dl>
        </div>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          At the default 0.5 cut-off the model calls most clean images stego: it flagged {p1a.test.confusion_matrix[0][1].toLocaleString("en-US")} of{" "}
          {(p1a.test.confusion_matrix[0][0] + p1a.test.confusion_matrix[0][1]).toLocaleString("en-US")} clean test images. This is why the
          application reports a percentile against clean reference images instead of using 0.5 as a verdict threshold.
        </p>
      </Card>

      <Card>
        <CardHeading eyebrow="Phase 1A" title="Payload sensitivity" />
        <div className="grid gap-6 md:grid-cols-2 md:items-start">
          <PayloadChart />
          <DataTable
            caption="Frozen-model metrics by payload"
            head={
              <>
                <Th>Payload (bpp)</Th>
                <Th>ROC-AUC</Th>
                <Th>F1</Th>
                <Th>Recall</Th>
              </>
            }
          >
            {research.payload_sensitivity.results.map((r) => (
              <tr key={r.payload_bpp}>
                <Th scope="row" className="font-mono text-fg">{r.payload_bpp.toFixed(2)}</Th>
                <Td>{fixed(r.roc_auc)}</Td>
                <Td>{fixed(r.f1)}</Td>
                <Td>{fixed(r.recall)}</Td>
              </tr>
            ))}
          </DataTable>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-muted">{research.payload_sensitivity.note}</p>
      </Card>

      <Card>
        <CardHeading eyebrow="Phase 1B" title="External validation on ALASKA2" />
        <p className="mb-3 text-sm leading-relaxed text-muted">
          ALASKA2 is a JPEG corpus (RGB; quality factors 75, 90 and 95). {p1b.n_source_groups} complete source groups ({p1b.n_files} images,
          about 50 sources per quality factor), evaluated with the frozen model and never used for training. The sample is small, so the
          intervals are wide. ROC-AUC with a 95% bootstrap interval over sources:
        </p>
        <DataTable
          caption="ALASKA2 ROC-AUC by embedding method"
          head={
            <>
              <Th>Embedding</Th>
              <Th>ROC-AUC</Th>
              <Th>95% interval</Th>
            </>
          }
        >
          {Object.entries(p1b.external).map(([key, v]) => (
            <tr key={key}>
              <Th scope="row" className="font-medium text-fg">{EXTERNAL_LABEL[key] ?? key}</Th>
              <Td>{fixed(v.roc_auc)}</Td>
              <Td>{interval(v.roc_auc_ci95)}</Td>
            </tr>
          ))}
        </DataTable>
        <ul className="mt-4 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-fg/90">
          <li>A small signal from our own controlled LSB embedding carries over to ALASKA2.</li>
          <li>
            The detector shows no measurable ability to separate JMiPOD, J-UNIWARD or UERD from cover (all three intervals include 0.5).
            That is a negative result about this pixel-domain detector, not about those methods.
          </li>
          <li>
            The share of clean covers scoring at or above 0.5 is {percent(p1b["clean_cover_fraction_score_ge_0.5"].alaska2)} on ALASKA2
            and {percent(p1b["clean_cover_fraction_score_ge_0.5"].bossbase_test)} on BOSSBase test: a property of the score
            distribution, not evidence of domain shift.
          </li>
        </ul>
      </Card>

      <Card>
        <CardHeading eyebrow="Phase 1B" title="Robustness to common transformations" />
        <p className="mb-3 text-sm leading-relaxed text-muted">
          The same transformation is applied to cover and stego images from the {p1b.robustness_n_test_sources.toLocaleString("en-US")} test
          sources. ΔAUC is the change from the untransformed AUC, with a 95% interval.
        </p>
        <DataTable
          caption="Change in ROC-AUC under image transformations"
          head={
            <>
              <Th>Transformation</Th>
              <Th>0.10 bpp: AUC</Th>
              <Th>ΔAUC [95%]</Th>
              <Th>0.40 bpp: AUC</Th>
              <Th>ΔAUC [95%]</Th>
            </>
          }
        >
          {p1b.robustness.map((row) => (
            <tr key={row.condition}>
              <Th scope="row" className="font-medium text-fg">{ROBUSTNESS_LABEL[row.condition] ?? row.condition}</Th>
              <Td>{fixed(row["0.10"].roc_auc)}</Td>
              <Td>
                {signed(row["0.10"].delta_auc)} <span className="text-muted">{interval(row["0.10"].delta_ci95)}</span>
              </Td>
              <Td>{fixed(row["0.40"].roc_auc)}</Td>
              <Td>
                {signed(row["0.40"].delta_auc)} <span className="text-muted">{interval(row["0.40"].delta_ci95)}</span>
              </Td>
            </tr>
          ))}
        </DataTable>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          JPEG re-compression and mild noise remove essentially all of the signal; a small crop and mild downscaling do not. Why
          resizing does not hurt has not been established.
        </p>
      </Card>

      <ResearchModelPanel />

      <Card>
        <CardHeading eyebrow="Scope" title="Limitations" />
        <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-fg/90">
          <li>Built for pixel-domain LSB replacement; it is not a general steganography detector.</li>
          <li>Weak in-domain (ROC-AUC {fixed(p1a.test.roc_auc)} at 0.10 bpp); not production-ready.</li>
          <li>Not evaluated on other embedding algorithms beyond ALASKA2&rsquo;s JMiPOD, J-UNIWARD and UERD, which it does not detect.</li>
          <li>Training data is synthetic: controlled LSB on one grayscale corpus. Results do not generalise automatically to other sources.</li>
          <li>ALASKA2 is one corpus with a small sample; camera-level source leakage could not be assessed.</li>
          <li>The score is uncalibrated and is not a probability; feature indicators are heuristic, not causal.</li>
          <li>A detection signal does not imply malicious intent, and a low score does not prove an image is clean.</li>
        </ul>
      </Card>
    </div>
  );
}
