"use client";
import { usePathname } from "next/navigation";
import React from "react";

interface Props {}

function Page(props: Props) {
  const {} = props;

  const pathName = usePathname();
  console.log(pathName);

  return (
    <div className="w-full min-h-screen flex justify-center items-center">
      <div>
        <h1>Welcome To Mudra setu!</h1>
      </div>
    </div>
  );
}

export default Page;
