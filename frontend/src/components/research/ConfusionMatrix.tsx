/** 2×2 confusion matrix; rows are the true class, columns the prediction. Layout: [[TN, FP], [FN, TP]]. */
export function ConfusionMatrix({ matrix, caption }: { matrix: number[][]; caption: string }) {
  const [[tn, fp], [fn, tp]] = matrix;
  const cell = "border border-line px-3 py-2 text-center font-mono text-sm tabular-nums";
  return (
    <table className="border-collapse text-sm">
      <caption className="mb-2 text-left text-xs text-muted">{caption}</caption>
      <thead>
        <tr>
          <td />
          <th scope="col" className="px-3 pb-1 text-xs font-normal text-muted">Predicted cover</th>
          <th scope="col" className="px-3 pb-1 text-xs font-normal text-muted">Predicted stego</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row" className="pr-3 text-right text-xs font-normal text-muted">Actual cover</th>
          <td className={cell}>{tn.toLocaleString("en-US")}</td>
          <td className={cell}>{fp.toLocaleString("en-US")}</td>
        </tr>
        <tr>
          <th scope="row" className="pr-3 text-right text-xs font-normal text-muted">Actual stego</th>
          <td className={cell}>{fn.toLocaleString("en-US")}</td>
          <td className={cell}>{tp.toLocaleString("en-US")}</td>
        </tr>
      </tbody>
    </table>
  );
}
