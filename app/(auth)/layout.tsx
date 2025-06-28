import React from "react";

interface Props {
  children: React.ReactNode;
}

function AuthLayout(props: Props) {
  const { children } = props;

  return (
    <div className="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex justify-center items-center min-h-screen pt-40">
      {children}
    </div>
  );
}

export default AuthLayout;
