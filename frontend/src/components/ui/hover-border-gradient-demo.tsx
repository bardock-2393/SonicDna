"use client";
import React from "react";
import { HoverBorderGradient } from "@/components/ui/hover-border-gradient";
import { Music } from "lucide-react";

export default function HoverBorderGradientDemo() {
  return (
    <div className="m-40 flex justify-center text-center">
      <HoverBorderGradient
        containerClassName="rounded-full"
        as="button"
        className="bg-surface text-text-primary flex items-center space-x-2"
      >
        <Music className="h-3 w-3" />
        <span>Lovable Music</span>
      </HoverBorderGradient>
    </div>
  );
}