export type CountMode = "rice" | "cone";

export type DetectionKind = "rice" | "cone";

export interface Detection {
  id: number;
  label: string;
  kind: DetectionKind;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export type DetectionStatus = "ok" | "no_detections" | "blurry" | "error";

export interface DetectionResponse {
  status: DetectionStatus;
  message: string;
  image_width: number;
  image_height: number;
  rice_count: number;
  cone_count: number;
  detections: Detection[];
}

export interface CapturedPhoto {
  uri: string;
  width: number;
  height: number;
}

// Navigation param list, shared between the stack navigator and screens so
// `navigation.navigate(...)` calls are type-checked.
export type RootStackParamList = {
  Home: undefined;
  Camera: { mode: CountMode };
  Processing: { mode: CountMode; photo: CapturedPhoto };
  Results: { mode: CountMode; photo: CapturedPhoto; result: DetectionResponse };
};
