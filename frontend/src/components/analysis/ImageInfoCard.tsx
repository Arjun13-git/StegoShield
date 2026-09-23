import { Card, CardHeading } from "@/components/ui/Card";
import type { ImageInfo } from "@/lib/types";
import { formatBytes } from "@/lib/utils";

export function ImageInfoCard({ image }: { image: ImageInfo }) {
  const rows: Array<[string, string]> = [
    ["Format", image.format],
    ["Dimensions", `${image.width} × ${image.height} px`],
    ["Original mode", image.original_mode],
    ["Analysed as", image.analyzed_as],
    ["File size", formatBytes(image.size_bytes)],
    ["Alpha channel", image.alpha_discarded ? "Discarded before analysis" : "None / not used"],
  ];
  return (
    <Card>
      <CardHeading eyebrow="Input" title="Image information" />
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted">{k}</dt>
            <dd className="text-fg">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}
