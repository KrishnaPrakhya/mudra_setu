import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const reset = searchParams.get('reset');

  if (reset === 'true') {
    try {
      const response = await fetch('http://localhost:8000/api/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`FastAPI server responded with status: ${response.status}`);
      }

      const result = await response.json();
      return NextResponse.json(result);
    } catch (error) {
      console.error('Error resetting sequence buffer:', error);
      return NextResponse.json(
        { error: 'Failed to reset sequence buffer' },
        { status: 500 }
      );
    }
  }

  // Handle prediction request
  try {
    const data = await request.json();
    const response = await fetch('http://localhost:8000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
        'Accept': 'application/json',
      },
      body: JSON.stringify(data),
      signal: AbortSignal.timeout(2000) // 2 second timeout
    });

    if (!response.ok) {
      throw new Error(`FastAPI server responded with status: ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error in predict API route:', error);
    return NextResponse.json(
      { error: 'Failed to process prediction request' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ status: 'ok' });
}