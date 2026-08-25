import { useCallback, useRef } from "react";
import { objectColor } from "./api";
import type { Point } from "./api";

export interface ViewerObject {
  id: number;
  points: Point[];
  maskUrl: string | null;
}

interface ViewerProps {
  imageUrl: string;
  /** Combined multi-object overlay from propagation (whole-frame PNG). */
  propagatedMaskUrl: string | null;
  objects: ViewerObject[];
  activeObjectId: number;
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
  propagatedMaskUrl,
  objects,
  activeObjectId,
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
      if (busy) return;
      const { col, row } = pickFromEvent(event, sliceWidth, sliceHeight);
      onPick(col, row, 1);
    },
    [onPick, sliceWidth, sliceHeight, busy],
  );

  const handleContextMenu = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (busy) return;
      const { col, row } = pickFromEvent(event, sliceWidth, sliceHeight);
      onPick(col, row, 0);
    },
    [onPick, sliceWidth, sliceHeight, busy],
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
      {propagatedMaskUrl && (
        <img
          className="viewer-layer viewer-mask"
          src={propagatedMaskUrl}
          alt=""
          style={{ opacity: maskOpacity }}
          draggable={false}
        />
      )}
      {objects.map(
        (obj) =>
          obj.maskUrl && (
            <img
              key={obj.id}
              className="viewer-layer viewer-mask"
              src={obj.maskUrl}
              alt=""
              style={{ opacity: maskOpacity }}
              draggable={false}
            />
          ),
      )}
      {objects.map((obj) =>
        obj.points.map((p, i) => (
          <span
            key={`${obj.id}-${i}`}
            className={`marker ${p.label === 1 ? "marker-pos" : "marker-neg"} ${
              obj.id === activeObjectId ? "marker-active" : ""
            }`}
            style={{
              left: p.col * scaleX,
              top: p.row * scaleY,
              backgroundColor: `${objectColor(obj.id)}99`,
            }}
          />
        )),
      )}
    </div>
  );
}
