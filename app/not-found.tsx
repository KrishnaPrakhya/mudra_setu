import React from "react";

interface Props {}

function NotFound(props: Props) {
  const {} = props;

  return (
    <div className="flex justify-center items-center min-h-screen">
      <h3>This Page Does not Exists ! Please Cooperate</h3>
    </div>
  );
}

export default NotFound;
