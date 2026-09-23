import { describe, expect, it } from "vitest";
import { checkImageFile, checkMessage, CLIENT_LIMITS, formatBytes, formatScore, riskLabel, shortHash, utf8ByteLength, verdictLabel } from "./utils";

describe("checkImageFile (advisory client-side validation)", () => {
  it("accepts PNG and JPEG within the limit", () => {
    expect(checkImageFile({ name: "a.png", type: "image/png", size: 1000 })).toBeNull();
    expect(checkImageFile({ name: "a.jpg", type: "image/jpeg", size: 1000 })).toBeNull();
  });

  it("rejects other types, empty files and oversized files", () => {
    expect(checkImageFile({ name: "a.gif", type: "image/gif", size: 1000 })).toMatch(/PNG and JPEG/);
    expect(checkImageFile({ name: "a.txt", type: "text/plain", size: 10 })).toMatch(/PNG and JPEG/);
    expect(checkImageFile({ name: "a.png", type: "image/png", size: 0 })).toMatch(/empty/);
    expect(checkImageFile({ name: "a.png", type: "image/png", size: CLIENT_LIMITS.maxUploadBytes + 1 })).toMatch(/usual limit/);
  });
});

describe("checkMessage", () => {
  it("requires a message", () => {
    expect(checkMessage("")).toMatch(/Enter a message/);
    expect(checkMessage("hi")).toBeNull();
  });

  it("counts UTF-8 bytes, not characters", () => {
    expect(utf8ByteLength("✓")).toBe(3);
    expect(checkMessage("a".repeat(CLIENT_LIMITS.maxMessageBytes))).toBeNull();
    expect(checkMessage("a".repeat(CLIENT_LIMITS.maxMessageBytes + 1))).toMatch(/longer than/);
    // 6000 three-byte characters = 18000 bytes > 16384 although only 6000 characters
    expect(checkMessage("✓".repeat(6000))).toMatch(/longer than/);
  });
});

describe("formatting", () => {
  it("formats bytes and scores", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatScore(52.5678)).toBe("52.57");
  });

  it("shortens a SHA-256 for display", () => {
    const h = "34707f83dc385e0846be3df7c56da930aad951a23fd3daa032a050449d02523d";
    expect(shortHash(h)).toBe("34707f83…02523d");
  });

  it("uses the required verdict and risk wording", () => {
    expect(verdictLabel("likely_cover")).toBe("Likely cover");
    expect(verdictLabel("potential_steganography")).toBe("Potential steganography");
    expect(verdictLabel("inconclusive")).toBe("Inconclusive");
    expect(riskLabel("not_assessed")).toBe("Not assessed");
    expect(verdictLabel("something_new")).toBe("Unknown");
  });
});
