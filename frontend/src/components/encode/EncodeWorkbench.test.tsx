import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClientError } from "@/lib/errors";
import { analysisFixture, modelInfoFixture, pngFile } from "@/test/fixtures";

vi.mock("@/lib/api", () => ({
  encodeImage: vi.fn(),
  analyzeImage: vi.fn(),
  getModelInfo: vi.fn(),
}));

import { analyzeImage, encodeImage, getModelInfo } from "@/lib/api";
import { EncodeWorkbench } from "./EncodeWorkbench";

const encodeMock = vi.mocked(encodeImage);
const analyzeMock = vi.mocked(analyzeImage);

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getModelInfo).mockResolvedValue(modelInfoFixture);
  globalThis.URL.createObjectURL = vi.fn(() => "blob:generated");
  globalThis.URL.revokeObjectURL = vi.fn();
});

async function chooseCover() {
  await userEvent.upload(screen.getByLabelText(/Drop a cover image here/), pngFile("cover.png"));
}

describe("EncodeWorkbench", () => {
  it("cannot submit without a cover image", () => {
    render(<EncodeWorkbench />);
    expect(screen.getByRole("button", { name: "Encode message" })).toBeDisabled();
    expect(screen.getByText("Choose a cover image first.")).toBeInTheDocument();
  });

  it("requires a message and does not call the API without one", async () => {
    render(<EncodeWorkbench />);
    await chooseCover();
    await userEvent.click(screen.getByRole("button", { name: "Encode message" }));
    expect(await screen.findByText("Enter a message to embed.")).toBeInTheDocument();
    expect(screen.getByLabelText(/Secret message/)).toHaveAttribute("aria-invalid", "true");
    expect(encodeMock).not.toHaveBeenCalled();
  });

  it("counts UTF-8 bytes in the message counter", async () => {
    render(<EncodeWorkbench />);
    await userEvent.type(screen.getByLabelText(/Secret message/), "✓✓");
    expect(screen.getByText(/6 \/ 16,384 bytes/)).toBeInTheDocument();
  });

  it("encodes, shows embedded/capacity from the headers, and offers a PNG download", async () => {
    encodeMock.mockResolvedValue({ blob: new Blob([new Uint8Array(10)], { type: "image/png" }), embeddedBytes: 14, capacityBytes: 6136, requestId: "enc-1" });
    render(<EncodeWorkbench />);
    await chooseCover();
    await userEvent.type(screen.getByLabelText(/Secret message/), "live check");
    await userEvent.click(screen.getByRole("button", { name: "Encode message" }));

    expect(await screen.findByText("Stego image generated")).toBeInTheDocument();
    expect(encodeMock).toHaveBeenCalledTimes(1);
    expect(encodeMock.mock.calls[0][1]).toBe("live check");
    expect(screen.getByText("14")).toBeInTheDocument();
    expect(screen.getByText("6,136")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Download PNG" });
    expect(link).toHaveAttribute("download", "stegoshield-encoded.png");
    expect(link).toHaveAttribute("href", "blob:generated");
    expect(screen.getByText(/not a secure channel/)).toBeInTheDocument();
  });

  it("shows a friendly error with the request id when the message does not fit", async () => {
    encodeMock.mockRejectedValue(
      new ApiClientError({ kind: "http", message: "The message does not fit in this image. Use a larger image or a shorter message.", status: 400, code: "payload_exceeds_capacity", requestId: "req-42" }),
    );
    render(<EncodeWorkbench />);
    await chooseCover();
    await userEvent.type(screen.getByLabelText(/Secret message/), "too long for this image");
    await userEvent.click(screen.getByRole("button", { name: "Encode message" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/does not fit in this image/);
    expect(alert).toHaveTextContent("req-42");
    expect(alert).toHaveTextContent("payload_exceeds_capacity");
    expect(screen.queryByText("Stego image generated")).not.toBeInTheDocument();
  });

  it("analyzes the generated image with the API and shows the result", async () => {
    encodeMock.mockResolvedValue({ blob: new Blob([new Uint8Array(10)]), embeddedBytes: 4, capacityBytes: 100, requestId: null });
    analyzeMock.mockResolvedValue(analysisFixture);
    render(<EncodeWorkbench />);
    await chooseCover();
    await userEvent.type(screen.getByLabelText(/Secret message/), "abcd");
    await userEvent.click(screen.getByRole("button", { name: "Encode message" }));
    await userEvent.click(await screen.findByRole("button", { name: "Analyze generated image" }));

    await waitFor(() => expect(analyzeMock).toHaveBeenCalledTimes(1));
    const sent = analyzeMock.mock.calls[0][0];
    expect(sent.name).toBe("stegoshield-encoded.png");
    expect(sent.type).toBe("image/png");
    expect(await screen.findByText("52.57")).toBeInTheDocument();
    expect(screen.getByText(/A single result is anecdotal/)).toBeInTheDocument();
  });
});
