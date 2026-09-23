import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClientError } from "@/lib/errors";
import { analysisFixture, modelInfoFixture, pngFile } from "@/test/fixtures";

vi.mock("@/lib/api", () => ({ analyzeImage: vi.fn(), getModelInfo: vi.fn() }));

import { analyzeImage, getModelInfo } from "@/lib/api";
import { AnalyzeWorkbench } from "./AnalyzeWorkbench";

const analyzeMock = vi.mocked(analyzeImage);

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getModelInfo).mockResolvedValue(modelInfoFixture);
  globalThis.URL.createObjectURL = vi.fn(() => "blob:preview");
  globalThis.URL.revokeObjectURL = vi.fn();
});

describe("AnalyzeWorkbench", () => {
  it("starts empty with the analyze action disabled", () => {
    render(<AnalyzeWorkbench />);
    expect(screen.getByText("No image selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze image" })).toBeDisabled();
  });

  it("uploads a selected image and renders the result while keeping the image selected", async () => {
    analyzeMock.mockResolvedValue(analysisFixture);
    render(<AnalyzeWorkbench />);
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), pngFile("cover.png"));
    expect(screen.getByText("cover.png")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));
    expect(await screen.findByText("52.57")).toBeInTheDocument();
    expect(analyzeMock).toHaveBeenCalledTimes(1);
    expect(analyzeMock.mock.calls[0][0].name).toBe("cover.png");
    expect(screen.getByText("cover.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze again" })).toBeEnabled();
  });

  it("shows a friendly error with the request id and lets the user retry", async () => {
    analyzeMock.mockRejectedValueOnce(
      new ApiClientError({ kind: "http", message: "The detection model is not available on the server right now.", status: 503, code: "model_unavailable", requestId: "req-503" }),
    );
    analyzeMock.mockResolvedValueOnce(analysisFixture);
    render(<AnalyzeWorkbench />);
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), pngFile());
    await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/model is not available/);
    expect(alert).toHaveTextContent("req-503");
    expect(alert).not.toHaveTextContent(/Traceback|Error:/);

    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("52.57")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("removes the image and clears the result", async () => {
    analyzeMock.mockResolvedValue(analysisFixture);
    render(<AnalyzeWorkbench />);
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), pngFile());
    await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));
    await screen.findByText("52.57");
    await userEvent.click(screen.getByRole("button", { name: /Remove cover\.png/ }));
    expect(screen.queryByText("52.57")).not.toBeInTheDocument();
    expect(screen.getByText("No image selected")).toBeInTheDocument();
  });
});
