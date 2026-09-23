import { redirect } from "next/navigation";

// Analyze is the primary experience; the root simply leads there.
export default function Home() {
  redirect("/analyze");
}
