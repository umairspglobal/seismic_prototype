import { useCallback, useRef } from "react";
import type { Point } from "./api";

interface ViewerProps {
  imageUrl: string;
  maskUrl: string | null;
  points: Point[];
  sliceWidth: number;
  sliceHeight: number;
  displayWidth: number;
  displayHeight: number;
  maskOpacity: number;
  busy: boolean;
  onPick: (col: number, row: number, label: 0 | 1) => void;
}

function pickFromEvent(
  event: React.MouseEvent<HTMLDivElement>,
  sliceWidth: number,
  sliceHeight: number,
): { col: number; row: number } {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.min(
    sliceWidth - 1,
    Math.max(0, Math.round((x / rect.width) * sliceWidth)),
  );
  const row = Math.min(
    sliceHeight - 1,
    Math.max(0, Math.round((y / rect.height) * sliceHeight)),
  );
  return { col, row };
}

/** Clickable seismic section with an instant marker + mask overlay stack. */
export function Viewer({
  imageUrl,
  maskUrl,
  points,
  sliceWidth,
  sliceHeight,
  displayWidth,
  displayHeight,
  maskOpacity,
  busy,
  onPick,
}: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // SAM 2-style prompts: left click = object, right click = background.
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const { col, row } = pickFromEvent(event, sliceWidth, sliceHeight);
      onPick(col, row, 1);
    },
    [onPick, sliceWidth, sliceHeight],
  );

  const handleContextMenu = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      const { col, row } = pickFromEvent(event, sliceWidth, sliceHeight);
      onPick(col, row, 0);
    },
    [onPick, sliceWidth, sliceHeight],
  );

  const scaleX = displayWidth / sliceWidth;
  const scaleY = displayHeight / sliceHeight;

  return (
    <div
      ref={containerRef}
      className={`viewer ${busy ? "viewer-busy" : ""}`}
      style={{ width: displayWidth, height: displayHeight }}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      <img className="viewer-layer" src={imageUrl} alt="Seismic section" draggable={false} />
      {maskUrl && (
        <img
          className="viewer-layer viewer-mask"
          src={maskUrl}
          alt=""
          style={{ opacity: maskOpacity }}
          draggable={false}
        />
      )}
      {points.map((p, i) => (
        <span
          key={i}
          className={`marker ${p.label === 1 ? "marker-pos" : "marker-neg"}`}
          style={{ left: p.col * scaleX, top: p.row * scaleY }}
        />
      ))}
    </div>
  );
}
