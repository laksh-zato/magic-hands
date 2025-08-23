import { onMount, createSignal } from "solid-js";
import { startCamera } from "../../utils/camera-utils";
import {
  createVideoGestureRecognizer,
  getHandData,
} from "../../utils/mediapipe-tasks-vision";
import type {
  GestureRecognizer,
  GestureRecognizerResult,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { Coords2D, triangleCentroid } from "../../utils/math-utils";
import { HAND_LABEL_INDEX } from "../../utils/mediapipe-hands-constants";
import { DataPanel } from "../DataPanel/DataPanel";
import { initFluidSimulation } from "./webgl-fluid-simulation";

import styles from "./MagicHands.module.scss";
import { createTouchEvent } from "../../utils/touch-utils";
import { Dialog } from "../Dialog/Dialog";
import classnames from "classnames";
import { MadeBy } from "../Branding/MadeBy";
import { ErrorAlert } from "./ErrorAlert";

// Gesture `categoryName`s: ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
const GESTURE_MATCH = "Open_Palm";
const FIST_GESTURE = "Closed_Fist";
const POINTING_GESTURE = "Pointing_Up";
const THUMB_UP_GESTURE = "Thumb_Up";

const LEFT_HAND_EVENT_ID = -10;
const RIGHT_HAND_EVENT_ID = -11;

let mainElement: HTMLDivElement;
let videoElement: HTMLVideoElement;
let fluidCanvasElement: HTMLCanvasElement;
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];

let previousTouches: TouchEvent[];

class MagicBall {
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: { r: number; g: number; b: number };
  life: number;
  maxLife: number;
  radius: number;
  
  constructor(x: number, y: number, vx: number, vy: number, color: { r: number; g: number; b: number }) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.color = color;
    this.maxLife = 120; // frames
    this.life = this.maxLife;
    this.radius = 0.03;
  }
  
  update() {
    // Physics
    this.x += this.vx * 0.016; // 60fps approximation
    this.y += this.vy * 0.016;
    this.vy += 0.5; // gravity
    
    // Bounce off screen edges
    if (this.x < 0 || this.x > 1) {
      this.vx *= -0.8;
      this.x = Math.max(0, Math.min(1, this.x));
    }
    
    // Reduce life
    this.life--;
    
    return this.life > 0;
  }
  
  createFluidEffect(fluidSimulation: ReturnType<typeof initFluidSimulation>) {
    const opacity = this.life / this.maxLife;
    const effectColor = {
      r: this.color.r * opacity * 0.8,
      g: this.color.g * opacity * 0.8,
      b: this.color.b * opacity * 0.8
    };
    
    // Create trailing effect
    for (let i = 0; i < 3; i++) {
      const angle = (i / 3) * Math.PI * 2;
      const offsetX = Math.cos(angle) * 0.01;
      const offsetY = Math.sin(angle) * 0.01;
      
      const x = this.x + offsetX;
      const y = this.y + offsetY;
      
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        const dx = this.vx * 20;
        const dy = this.vy * 20;
        fluidSimulation.createSplat?.(x, y, dx, dy, effectColor);
      }
    }
  }
}

let magicBalls: MagicBall[] = [];

interface HandVelocity {
  x: number;
  y: number;
  timestamp: number;
}

let leftHandHistory: HandVelocity[] = [];
let rightHandHistory: HandVelocity[] = [];
const VELOCITY_HISTORY_SIZE = 5;
const THROW_VELOCITY_THRESHOLD = 8;

function updateHandVelocityHistory(
  history: HandVelocity[],
  coords: { pageX: number; pageY: number }
) {
  const now = Date.now();
  const canvas = fluidCanvasElement;
  const pixelRatio = window.devicePixelRatio || 1;
  
  // Convert to normalized coordinates
  const normalizedX = coords.pageX * pixelRatio / canvas.width;
  const normalizedY = 1.0 - (coords.pageY * pixelRatio / canvas.height);
  
  history.push({
    x: normalizedX,
    y: normalizedY,
    timestamp: now
  });
  
  // Keep only recent history
  while (history.length > VELOCITY_HISTORY_SIZE) {
    history.shift();
  }
}

function detectThrowingMotion(history: HandVelocity[]): { vx: number; vy: number } | null {
  if (history.length < 3) return null;
  
  // Calculate average velocity over recent frames
  let avgVx = 0;
  let avgVy = 0;
  let validSamples = 0;
  
  for (let i = 1; i < history.length; i++) {
    const dt = (history[i].timestamp - history[i-1].timestamp) / 1000; // to seconds
    if (dt > 0 && dt < 0.1) { // reasonable time delta
      const vx = (history[i].x - history[i-1].x) / dt;
      const vy = (history[i].y - history[i-1].y) / dt;
      
      avgVx += vx;
      avgVy += vy;
      validSamples++;
    }
  }
  
  if (validSamples === 0) return null;
  
  avgVx /= validSamples;
  avgVy /= validSamples;
  
  const speed = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
  
  if (speed > THROW_VELOCITY_THRESHOLD) {
    return { vx: avgVx, vy: avgVy };
  }
  
  return null;
}

function throwMagicBall(
  handCoords: { pageX: number; pageY: number },
  velocity: { vx: number; vy: number },
  isLeftHand: boolean
) {
  const canvas = fluidCanvasElement;
  const pixelRatio = window.devicePixelRatio || 1;
  
  // Convert to normalized coordinates
  const startX = handCoords.pageX * pixelRatio / canvas.width;
  const startY = 1.0 - (handCoords.pageY * pixelRatio / canvas.height);
  
  // Generate magical color
  const hue = Math.random();
  const color = {
    r: 0.3 + Math.random() * 0.7,
    g: 0.3 + Math.random() * 0.7,
    b: 0.3 + Math.random() * 0.7
  };
  
  // Create new magic ball with throwing velocity
  const ball = new MagicBall(startX, startY, velocity.vx, velocity.vy, color);
  magicBalls.push(ball);
  
  // Limit number of balls to prevent performance issues
  while (magicBalls.length > 10) {
    magicBalls.shift();
  }
}

function updateMagicBalls(fluidSimulation: ReturnType<typeof initFluidSimulation>) {
  for (let i = magicBalls.length - 1; i >= 0; i--) {
    const ball = magicBalls[i];
    const stillAlive = ball.update();
    
    if (stillAlive) {
      ball.createFluidEffect(fluidSimulation);
    } else {
      magicBalls.splice(i, 1);
    }
  }
}

function getHandCentroid(landmarks?: NormalizedLandmark[]) {
  if (!landmarks) {
    return;
  }

  const wristCoords = landmarks[HAND_LABEL_INDEX.WRIST];
  const indexMCPCoords = landmarks[HAND_LABEL_INDEX.INDEX_FINGER_MCP];
  const pinkyMCPCoords = landmarks[HAND_LABEL_INDEX.PINKY_FINGER_MCP];
  return triangleCentroid(wristCoords, indexMCPCoords, pinkyMCPCoords);
}

function getTouch({
  pageX,
  pageY,
  identifier,
}: {
  pageX: number;
  pageY: number;
  identifier: number;
}) {
  return {
    identifier,
    pageX,
    pageY,
    screenX: pageX,
    screenY: pageY,
    clientX: pageX,
    clientY: pageY,
  };
}

function getPageCoords(normalizedCoord?: Coords2D) {
  if (!normalizedCoord) {
    return;
  }
  const pixelRatio = window.devicePixelRatio || 1;
  const screenWidth = fluidCanvasElement.width;
  const screenHeight = fluidCanvasElement.height;

  // Convert to screen cooordinates
  const pageX = (normalizedCoord.x * screenWidth) / pixelRatio;
  const pageY = (normalizedCoord.y * screenHeight) / pixelRatio;

  return {
    pageX,
    pageY,
  };
}

function getEvent(touches: TouchEvent[]) {
  if (!previousTouches) {
    previousTouches = [];
  }

  if (previousTouches.length && previousTouches.length >= touches.length) {
    return createTouchEvent({
      type: "touchmove",
      touches,
    });
  } else {
    return createTouchEvent({
      type: "touchstart",
      touches,
    });
  }
}

function createBothHandsGlowEffect(
  fluidSimulation: ReturnType<typeof initFluidSimulation>,
  leftHandCoords: { pageX: number; pageY: number },
  rightHandCoords: { pageX: number; pageY: number }
) {
  const canvas = fluidCanvasElement;
  
  // Convert page coordinates to normalized coordinates (0-1)
  const pixelRatio = window.devicePixelRatio || 1;
  const leftX = leftHandCoords.pageX * pixelRatio / canvas.width;
  const leftY = 1.0 - (leftHandCoords.pageY * pixelRatio / canvas.height);
  const rightX = rightHandCoords.pageX * pixelRatio / canvas.width;
  const rightY = 1.0 - (rightHandCoords.pageY * pixelRatio / canvas.height);
  const centerNormX = 0.5;
  const centerNormY = 0.5;
  
  // Create gentle continuous streams with much lower force
  const numSplats = 2;
  
  for (let i = 0; i < numSplats; i++) {
    const t = (i + 1) / (numSplats + 1);
    
    // Left hand to center - Blue/Cyan color with moderate force
    const leftInterpX = leftX + (centerNormX - leftX) * t;
    const leftInterpY = leftY + (centerNormY - leftY) * t;
    const leftDx = (centerNormX - leftX) * 200; // Increased force for visibility
    const leftDy = (centerNormY - leftY) * 200;
    const leftColor = { r: 0.05, g: 0.15, b: 0.3 }; // More visible blue
    
    fluidSimulation.createSplat?.(leftInterpX, leftInterpY, leftDx, leftDy, leftColor);
    
    // Right hand to center - Red/Orange color with moderate force
    const rightInterpX = rightX + (centerNormX - rightX) * t;
    const rightInterpY = rightY + (centerNormY - rightY) * t;
    const rightDx = (centerNormX - rightX) * 200; // Increased force for visibility
    const rightDy = (centerNormY - rightY) * 200;
    const rightColor = { r: 0.3, g: 0.1, b: 0.05 }; // More visible red/orange
    
    fluidSimulation.createSplat?.(rightInterpX, rightInterpY, rightDx, rightDy, rightColor);
  }
}

function createBothPointingRaysToCenter(
  fluidSimulation: ReturnType<typeof initFluidSimulation>,
  leftHandCoords: { pageX: number; pageY: number },
  rightHandCoords: { pageX: number; pageY: number }
) {
  const canvas = fluidCanvasElement;
  const pixelRatio = window.devicePixelRatio || 1;
  
  // Convert to normalized coordinates
  const leftX = leftHandCoords.pageX * pixelRatio / canvas.width;
  const leftY = 1.0 - (leftHandCoords.pageY * pixelRatio / canvas.height);
  const rightX = rightHandCoords.pageX * pixelRatio / canvas.width;
  const rightY = 1.0 - (rightHandCoords.pageY * pixelRatio / canvas.height);
  const centerX = 0.5;
  const centerY = 0.5;
  
  const time = Date.now() * 0.002; // Slow animation
  const numRayPoints = 3; // Points along each ray
  
  // Left hand ray to center
  const leftDirX = centerX - leftX;
  const leftDirY = centerY - leftY;
  const leftLength = Math.sqrt(leftDirX * leftDirX + leftDirY * leftDirY);
  
  if (leftLength > 0) {
    const leftNormX = leftDirX / leftLength;
    const leftNormY = leftDirY / leftLength;
    
    for (let i = 1; i <= numRayPoints; i++) {
      const t = i / (numRayPoints + 1);
      const distance = leftLength * t;
      
      // Add jiggle
      const jiggle = Math.sin(time + i) * 0.008;
      const perpX = -leftNormY;
      const perpY = leftNormX;
      
      const x = leftX + leftNormX * distance + perpX * jiggle;
      const y = leftY + leftNormY * distance + perpY * jiggle;
      
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        const dx = leftNormX * 40;
        const dy = leftNormY * 40;
        
        // Random bright colors for left ray
        const hue = Math.random();
        const color = {
          r: 0.2 + Math.random() * 0.3,
          g: 0.1 + Math.random() * 0.2,
          b: 0.3 + Math.random() * 0.3
        };
        
        fluidSimulation.createSplat?.(x, y, dx, dy, color);
      }
    }
  }
  
  // Right hand ray to center
  const rightDirX = centerX - rightX;
  const rightDirY = centerY - rightY;
  const rightLength = Math.sqrt(rightDirX * rightDirX + rightDirY * rightDirY);
  
  if (rightLength > 0) {
    const rightNormX = rightDirX / rightLength;
    const rightNormY = rightDirY / rightLength;
    
    for (let i = 1; i <= numRayPoints; i++) {
      const t = i / (numRayPoints + 1);
      const distance = rightLength * t;
      
      // Add jiggle
      const jiggle = Math.sin(time + i + Math.PI) * 0.008; // Phase offset
      const perpX = -rightNormY;
      const perpY = rightNormX;
      
      const x = rightX + rightNormX * distance + perpX * jiggle;
      const y = rightY + rightNormY * distance + perpY * jiggle;
      
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        const dx = rightNormX * 40;
        const dy = rightNormY * 40;
        
        // Random bright colors for right ray
        const hue = Math.random();
        const color = {
          r: 0.3 + Math.random() * 0.3,
          g: 0.2 + Math.random() * 0.2,
          b: 0.1 + Math.random() * 0.2
        };
        
        fluidSimulation.createSplat?.(x, y, dx, dy, color);
      }
    }
  }
}

function createWigglyBall(
  fluidSimulation: ReturnType<typeof initFluidSimulation>,
  handCoords: { pageX: number; pageY: number },
  isLeftHand: boolean
) {
  const canvas = fluidCanvasElement;
  const pixelRatio = window.devicePixelRatio || 1;
  const centerX = handCoords.pageX * pixelRatio / canvas.width;
  const centerY = 1.0 - (handCoords.pageY * pixelRatio / canvas.height);
  
  const time = Date.now() * 0.003; // Slower animation
  const numPoints = 6; // Fewer points for subtler effect
  const baseRadius = 0.04; // Much smaller radius
  
  for (let i = 0; i < numPoints; i++) {
    const angle = (i / numPoints) * Math.PI * 2;
    const wiggle = Math.sin(time + angle * 2) * 0.008; // Subtle wiggle
    const radius = baseRadius + wiggle;
    
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    
    // Very gentle inward force
    const dx = (centerX - x) * 30;
    const dy = (centerY - y) * 30;
    
    // More fantasy-like gentle colors
    const color = isLeftHand 
      ? { r: 0.08, g: 0.02, b: 0.12 } // Gentle purple/magenta
      : { r: 0.12, g: 0.06, b: 0.02 }; // Gentle orange/amber
    
    fluidSimulation.createSplat?.(x, y, dx, dy, color);
  }
}

function createFingerRay(
  fluidSimulation: ReturnType<typeof initFluidSimulation>,
  landmarks: NormalizedLandmark[],
  handCoords: { pageX: number; pageY: number },
  isLeftHand: boolean
) {
  const canvas = fluidCanvasElement;
  const pixelRatio = window.devicePixelRatio || 1;
  
  // Get finger tip and MCP (base) positions for index finger
  const fingerTip = landmarks[HAND_LABEL_INDEX.INDEX_FINGER_TIP];
  const fingerMCP = landmarks[HAND_LABEL_INDEX.INDEX_FINGER_MCP];
  
  if (!fingerTip || !fingerMCP) return;
  
  // Convert to page coordinates
  const tipPageCoords = getPageCoords(fingerTip);
  const mcpPageCoords = getPageCoords(fingerMCP);
  
  if (!tipPageCoords || !mcpPageCoords) return;
  
  // Calculate direction vector from MCP to tip
  const directionX = tipPageCoords.pageX - mcpPageCoords.pageX;
  const directionY = tipPageCoords.pageY - mcpPageCoords.pageY;
  
  // Normalize direction
  const length = Math.sqrt(directionX * directionX + directionY * directionY);
  if (length === 0) return;
  
  const normalizedDirX = directionX / length;
  const normalizedDirY = directionY / length;
  
  // Convert to normalized coordinates
  const startX = tipPageCoords.pageX * pixelRatio / canvas.width;
  const startY = 1.0 - (tipPageCoords.pageY * pixelRatio / canvas.height);
  
  // Create a slow, thin jiggly ray stream
  const rayLength = 0.2; // Even shorter ray
  const numRayPoints = 2; // Very few points for minimal effect
  const time = Date.now() * 0.001; // Much slower animation
  
  for (let i = 1; i <= numRayPoints; i++) {
    const t = i / numRayPoints;
    const distance = rayLength * t;
    
    // Add very subtle jiggle perpendicular to ray direction
    const jiggleAmount = Math.sin(time + i) * 0.005;
    const perpX = -normalizedDirY; // Perpendicular to direction
    const perpY = normalizedDirX;
    
    const x = startX + (normalizedDirX * distance * pixelRatio / canvas.width) + (perpX * jiggleAmount);
    const y = startY - (normalizedDirY * distance * pixelRatio / canvas.height) + (perpY * jiggleAmount);
    
    // Keep ray within bounds
    if (x < 0 || x > 1 || y < 0 || y > 1) break;
    
    // Very small force to prevent screen filling
    const forceMultiplier = 20; // Much smaller force
    const dx = normalizedDirX * forceMultiplier;
    const dy = -normalizedDirY * forceMultiplier;
    
    // Random gentle colors
    const hue = Math.random(); // Random hue
    const saturation = 0.6 + Math.random() * 0.4; // High saturation
    const value = 0.3 + Math.random() * 0.2; // Moderate brightness
    
    // Convert HSV to RGB
    const c = value * saturation;
    const x_color = c * (1 - Math.abs((hue * 6) % 2 - 1));
    const m = value - c;
    
    let r, g, b;
    if (hue < 1/6) { r = c; g = x_color; b = 0; }
    else if (hue < 2/6) { r = x_color; g = c; b = 0; }
    else if (hue < 3/6) { r = 0; g = c; b = x_color; }
    else if (hue < 4/6) { r = 0; g = x_color; b = c; }
    else if (hue < 5/6) { r = x_color; g = 0; b = c; }
    else { r = c; g = 0; b = x_color; }
    
    const color = { r: r + m, g: g + m, b: b + m };
    
    fluidSimulation.createSplat?.(x, y, dx, dy, color);
  }
}

function processResults({
  results,
  fluidSimulation,
  isFlipped,
}: {
  results: GestureRecognizerResult;
  fluidSimulation: ReturnType<typeof initFluidSimulation>;
  isFlipped?: boolean;
}) {
  const currentHandData = getHandData({ results, isFlipped });

  const touches: TouchEvent[] = [];
  let leftHandCoords: ReturnType<typeof getPageCoords> | undefined;
  let rightHandCoords: ReturnType<typeof getPageCoords> | undefined;
  let leftFistCoords: ReturnType<typeof getPageCoords> | undefined;
  let rightFistCoords: ReturnType<typeof getPageCoords> | undefined;
  let leftPointingCoords: ReturnType<typeof getPageCoords> | undefined;
  let rightPointingCoords: ReturnType<typeof getPageCoords> | undefined;
  let leftThumbUpCoords: ReturnType<typeof getPageCoords> | undefined;
  let rightThumbUpCoords: ReturnType<typeof getPageCoords> | undefined;

  // Check for open palm gestures
  if (currentHandData?.left?.gesture?.categoryName === GESTURE_MATCH) {
    const handMiddleCoords = getHandCentroid(currentHandData?.left?.landmarks);
    leftHandCoords = getPageCoords(handMiddleCoords);

    if (leftHandCoords?.pageX && leftHandCoords?.pageY) {
      const touch = getTouch({
        pageX: leftHandCoords.pageX,
        pageY: leftHandCoords.pageY,
        identifier: LEFT_HAND_EVENT_ID,
      });
      touches.push(touch as any);
    }
  }
  if (currentHandData?.right?.gesture?.categoryName === GESTURE_MATCH) {
    const handMiddleCoords = getHandCentroid(currentHandData?.right?.landmarks);
    rightHandCoords = getPageCoords(handMiddleCoords);

    if (rightHandCoords?.pageX && rightHandCoords?.pageY) {
      const touch = getTouch({
        pageX: rightHandCoords.pageX,
        pageY: rightHandCoords.pageY,
        identifier: RIGHT_HAND_EVENT_ID,
      });
      touches.push(touch as any);
    }
  }

  // Check for fist gestures
  if (currentHandData?.left?.gesture?.categoryName === FIST_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.left?.landmarks);
    leftFistCoords = getPageCoords(handMiddleCoords);
  }
  if (currentHandData?.right?.gesture?.categoryName === FIST_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.right?.landmarks);
    rightFistCoords = getPageCoords(handMiddleCoords);
  }

  // Check for pointing gestures
  if (currentHandData?.left?.gesture?.categoryName === POINTING_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.left?.landmarks);
    leftPointingCoords = getPageCoords(handMiddleCoords);
  }
  if (currentHandData?.right?.gesture?.categoryName === POINTING_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.right?.landmarks);
    rightPointingCoords = getPageCoords(handMiddleCoords);
  }

  // Check for thumb up gestures (for throwing magic balls)
  if (currentHandData?.left?.gesture?.categoryName === THUMB_UP_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.left?.landmarks);
    leftThumbUpCoords = getPageCoords(handMiddleCoords);
    
    if (leftThumbUpCoords) {
      updateHandVelocityHistory(leftHandHistory, leftThumbUpCoords);
      const throwVelocity = detectThrowingMotion(leftHandHistory);
      if (throwVelocity) {
        throwMagicBall(leftThumbUpCoords, throwVelocity, true);
        leftHandHistory.length = 0; // Clear history to prevent multiple throws
      }
    }
  }
  if (currentHandData?.right?.gesture?.categoryName === THUMB_UP_GESTURE) {
    const handMiddleCoords = getHandCentroid(currentHandData?.right?.landmarks);
    rightThumbUpCoords = getPageCoords(handMiddleCoords);
    
    if (rightThumbUpCoords) {
      updateHandVelocityHistory(rightHandHistory, rightThumbUpCoords);
      const throwVelocity = detectThrowingMotion(rightHandHistory);
      if (throwVelocity) {
        throwMagicBall(rightThumbUpCoords, throwVelocity, false);
        rightHandHistory.length = 0; // Clear history to prevent multiple throws
      }
    }
  }

  // Handle different gesture combinations
  if (leftFistCoords?.pageX && leftFistCoords?.pageY) {
    createWigglyBall(fluidSimulation, leftFistCoords, true);
  }
  if (rightFistCoords?.pageX && rightFistCoords?.pageY) {
    createWigglyBall(fluidSimulation, rightFistCoords, false);
  }

  // Handle pointing gestures - create rays
  // If both hands are pointing, create dual rays to center
  if (leftPointingCoords?.pageX && leftPointingCoords?.pageY && rightPointingCoords?.pageX && rightPointingCoords?.pageY) {
    createBothPointingRaysToCenter(fluidSimulation, leftPointingCoords, rightPointingCoords);
  } else {
    // Single hand pointing rays
    if (leftPointingCoords?.pageX && leftPointingCoords?.pageY && currentHandData?.left?.landmarks) {
      createFingerRay(fluidSimulation, currentHandData.left.landmarks, leftPointingCoords, true);
    }
    if (rightPointingCoords?.pageX && rightPointingCoords?.pageY && currentHandData?.right?.landmarks) {
      createFingerRay(fluidSimulation, currentHandData.right.landmarks, rightPointingCoords, false);
    }
  }

  // If both hands are open palm, create continuous glow from both sides to center
  if (leftHandCoords && rightHandCoords && leftHandCoords.pageX && leftHandCoords.pageY && rightHandCoords.pageX && rightHandCoords.pageY) {
    createBothHandsGlowEffect(fluidSimulation, leftHandCoords, rightHandCoords);
  } else if (touches.length > 0) {
    // Regular touch handling for single open palm
    const event = getEvent(touches);
    if (event.type === "touchstart") {
      fluidSimulation.sendTouchStart(event as TouchEvent);
    } else if (event.type === "touchmove") {
      fluidSimulation.sendTouchMove(event as TouchEvent);
    } else if (event.type === "touchend") {
      fluidSimulation.sendTouchEnd(event as TouchEvent);
    }
  }

  // Update magic balls physics and effects
  updateMagicBalls(fluidSimulation);

  previousTouches = touches;
}

export const MagicHands = () => {
  const isDebug =
    new URLSearchParams(window.location.search).get("debug") === "true";
  const [gestureResults, setGestureResults] =
    createSignal<GestureRecognizerResult>();
  const [showDialog, setShowDialog] = createSignal(false);
  const [hasMounted, setHasMounted] = createSignal(false);
  const [error, setError] = createSignal<string>();
  const [isRecording, setIsRecording] = createSignal(false);
  const hasError = () => hasMounted() && error();

  onMount(async () => {
    setHasMounted(true);
    const fluidSimulation = initFluidSimulation({
      canvasEl: fluidCanvasElement,
    });

    let gestureRecognizer: GestureRecognizer;
    try {
      gestureRecognizer = await createVideoGestureRecognizer();
    } catch (e: any) {
      const errorType = "Error initialising gesture recognizer.";
      setError(errorType);
      console.error(e);
      if ((globalThis as any).plausible) {
        (globalThis as any).plausible("Error", {
          props: { type: errorType, error: e.toString() },
        });
      }
    }
    const processWebcam = async () => {
      videoElement.classList.add(styles.initialised);

      if (!gestureRecognizer) {
        return;
      }

      let nowInMs = Date.now();
      const results = gestureRecognizer.recognizeForVideo(
        videoElement,
        nowInMs
      );

      processResults({ results, fluidSimulation, isFlipped: true });
      setGestureResults(results);

      // Continue processing webcam
      requestAnimationFrame(processWebcam);
    };

    let videoStream: MediaStream;
    startCamera({
      videoElement,
      onLoad: () => {
        videoStream = videoElement.srcObject as MediaStream;
        processWebcam();
      },
    });

    // Video recording functions
    const startRecording = () => {
      if (!videoStream || isRecording()) return;
      
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(videoStream, {
        mimeType: 'video/webm;codecs=vp9'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `magic-hands-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setIsRecording(false);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    };
    
    const stopRecording = () => {
      if (mediaRecorder && isRecording()) {
        mediaRecorder.stop();
      }
    };

    // Expose recording functions globally for the button
    (window as any).recordingControls = { startRecording, stopRecording };
  });

  return (
    <main ref={mainElement}>
      {hasError() && <ErrorAlert error={error()!} />}
      {isDebug && <DataPanel results={gestureResults()} />}

      <video
        ref={videoElement}
        id={styles.inputVideo}
        autoplay
        playsinline
      ></video>
      <canvas ref={fluidCanvasElement} id={styles.fluidCanvas}></canvas>

      <button
        class={classnames("unstyledButton", styles.infoButton, {
          [styles.hide]: showDialog(),
        })}
        onClick={() => setShowDialog(true)}
      >
        <i class="ph ph-info"></i>
      </button>

      <button
        class={classnames("unstyledButton", styles.recordButton, {
          [styles.recording]: isRecording(),
        })}
        onClick={() => {
          const controls = (window as any).recordingControls;
          if (controls) {
            if (isRecording()) {
              controls.stopRecording();
            } else {
              controls.startRecording();
            }
          }
        }}
        title={isRecording() ? "Stop Recording" : "Start Recording"}
      >
        <i class={isRecording() ? "ph ph-stop-circle-fill" : "ph ph-record-fill"}></i>
      </button>

      

      <footer class={classnames(styles.footer)}>
        <MadeBy />
      </footer>
    </main>
  );
};
