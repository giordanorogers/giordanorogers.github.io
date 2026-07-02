export function formatDate(d: Date): string {
  return d
    .toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
    .replace(',', '')
    .toUpperCase();
}

export function readingTime(body: string): number {
  return Math.max(1, Math.round(body.trim().split(/\s+/).length / 220));
}
