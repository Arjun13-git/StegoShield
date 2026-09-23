import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ModelInfoState } from "@/hooks/useModelInfo";
import type { AnalysisResponse } from "@/lib/types";
import { analysisFixture, modelInfoFixture } from "@/test/fixtures";
import { AnalysisResult } from "./AnalysisResult";

const loaded: ModelInfoState = { status: "ready", info: modelInfoFixture };

function renderResult(overrides: Partial<AnalysisResponse> = {}, modelInfo: ModelInfoState = loaded) {
  return render(<AnalysisResult analysis={{ ...analysisFixture, ...overrides }} modelInfo={modelInfo} />);
}

describe("AnalysisResult", () => {
  it("shows the score as an uncalibrated model score, never as a percentage or probability", () => {
    const { container } = renderResult();
    expect(screen.getByText("52.57")).toBeInTheDocument();
    expect(screen.getByText("Stego score")).toBeInTheDocument();
    expect(screen.getByText(/Uncalibrated model score/)).toBeInTheDocument();
    expect(screen.getByText(/Not a probability of hidden data/)).toBeInTheDocument();
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/52\.57\s*%/);
    expect(text).not.toMatch(/chance of steganography/i);
    expect(text).not.toMatch(/definitely|confirmed|malicious image/i);
  });

  it("uses the backend's risk level and verdict with the required wording", () => {
    renderResult();
    const list = screen.getByText("Risk level").closest("dl") as HTMLElement;
    expect(within(list).getByText("Elevated")).toBeInTheDocument();
    expect(within(list).getByText("Potential steganography")).toBeInTheDocument();
    expect(within(list).getByText("91.2")).toBeInTheDocument();
  });

  it.each([
    ["low", "likely_cover", "Low", "Likely cover"],
    ["high", "potential_steganography", "High", "Potential steganography"],
    ["not_assessed", "inconclusive", "Not assessed", "Inconclusive"],
  ] as const)("renders risk %s / verdict %s", (risk, verdict, riskText, verdictText) => {
    renderResult({ risk_level: risk, verdict });
    const list = screen.getByText("Risk level").closest("dl") as HTMLElement;
    expect(within(list).getByText(riskText)).toBeInTheDocument();
    expect(within(list).getByText(verdictText)).toBeInTheDocument();
  });

  it("explains a missing reference instead of inventing a percentile", () => {
    renderResult({ risk_level: "not_assessed", verdict: "inconclusive", cover_reference_percentile: null });
    expect(screen.getByText("Not available")).toBeInTheDocument();
    expect(screen.getByText(/cannot be placed on a scale/)).toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });

  it("marks the percentile meter accessibly and draws the risk-level marks only from model-info", () => {
    renderResult();
    const meter = screen.getByRole("meter", { name: /reference clean-image scores/i });
    expect(meter).toHaveAttribute("aria-valuenow", "91.2");
    expect(screen.getByText(/elevated from the 90th percentile, high from the 99th percentile/)).toBeInTheDocument();
  });

  it("omits the risk-level marks when model information is unavailable", () => {
    renderResult({}, { status: "error", message: "down", requestId: null });
    expect(screen.queryByText(/from the 90th percentile/)).not.toBeInTheDocument();
    expect(screen.getByText(/Model information is unavailable: down/)).toBeInTheDocument();
  });

  it("presents indicators as heuristic evidence with the backend's fields", () => {
    renderResult();
    expect(screen.getByRole("heading", { name: "Heuristic model evidence" })).toBeInTheDocument();
    expect(screen.getByText(/not proof, not a\s+causal explanation/)).toBeInTheDocument();
    expect(screen.getByText("LSB transition rate (blue channel)")).toBeInTheDocument();
    expect(screen.getByText("lsb_transition_b")).toBeInTheDocument();
    expect(screen.getByText("+0.72 σ")).toBeInTheDocument();
    expect(screen.getByText("0.1085")).toBeInTheDocument();
    expect(screen.getByText(/Deviates like the LSB-stego reference/)).toBeInTheDocument();
    expect(screen.getByText(analysisFixture.explanation_method)).toBeInTheDocument();
  });

  it("shows warnings distinctly from limitations, and limitations are visible by default", () => {
    renderResult();
    expect(screen.getByText(/not 512x512/)).toBeInTheDocument();
    expect(screen.getByText("Input caveat")).toBeInTheDocument();
    const details = screen.getByText("What this detector cannot tell you").closest("details") as HTMLDetailsElement;
    expect(details.open).toBe(true);
    expect(within(details).getByText(/JMiPOD, J-UNIWARD, UERD/)).toBeInTheDocument();
  });

  it("renders image and model information, with a shortened, copyable model hash", () => {
    renderResult();
    expect(screen.getByText("256 × 256 px")).toBeInTheDocument();
    expect(screen.getByText("RGB (grey replicated)")).toBeInTheDocument();
    expect(screen.getByText("34707f83…02523d")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy full model SHA-256" })).toBeInTheDocument();
  });
});
