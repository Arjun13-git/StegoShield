export function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto w-full max-w-6xl px-4 py-5 text-xs leading-relaxed text-muted sm:px-6">
        <p>
          StegoShield is an educational research prototype. It targets pixel-domain LSB steganography, is a weak detector on the
          data it was tested on, and does not detect JPEG-domain methods. A score is not proof that hidden data is present or
          absent, and says nothing about intent.
        </p>
      </div>
    </footer>
  );
}
