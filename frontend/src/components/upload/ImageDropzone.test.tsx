import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { pngFile } from "@/test/fixtures";
import { ImageDropzone } from "./ImageDropzone";

describe("ImageDropzone", () => {
  it("passes an acceptable PNG to onSelect", async () => {
    const onSelect = vi.fn();
    render(<ImageDropzone onSelect={onSelect} />);
    const file = pngFile();
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), file);
    expect(onSelect).toHaveBeenCalledWith(file);
  });

  it("rejects an unsupported type with an alert and does not call onSelect", async () => {
    const onSelect = vi.fn();
    render(<ImageDropzone onSelect={onSelect} />);
    const gif = new File(["x"], "a.gif", { type: "image/gif" });
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), gif, { applyAccept: false });
    expect(await screen.findByRole("alert")).toHaveTextContent(/Only PNG and JPEG/);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("rejects an empty file", async () => {
    const onSelect = vi.fn();
    render(<ImageDropzone onSelect={onSelect} />);
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), new File([], "e.png", { type: "image/png" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/empty/);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("rejects an oversized file (advisory; the server is authoritative)", async () => {
    const onSelect = vi.fn();
    render(<ImageDropzone onSelect={onSelect} />);
    const big = pngFile("big.png", 10 * 1024 * 1024 + 1);
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), big);
    expect(await screen.findByRole("alert")).toHaveTextContent(/usual limit/);
    expect(onSelect).not.toHaveBeenCalled();
  });
});
