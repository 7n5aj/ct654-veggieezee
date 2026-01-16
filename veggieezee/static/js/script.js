// Vegetable data
const vegetables = [
  {
    name: "Green Beans",
    localName: "Simi",
    category: "Legumes",
    price: 85.38,
    unit: "/piece",
    change: 3.6,
  },
  {
    name: "Onion",
    localName: "Pyaaj",
    category: "Alliums",
    price: 50.49,
    unit: "/kg",
    change: -2.8,
  },
  {
    name: "Potato",
    localName: "Aalu",
    category: "Root Vegetables",
    price: 165.29,
    unit: "/kg",
    change: -4.1,
  },
  {
    name: "Green Beans",
    localName: "Simi",
    category: "Legumes",
    price: 76.89,
    unit: "/kg",
    change: -2.8,
  },
  {
    name: "Onion",
    localName: "Pyaaj",
    category: "Alliums",
    price: 42.56,
    unit: "/kg",
    change: -8.0,
  },
  {
    name: "Potato",
    localName: "Aalu",
    category: "Root Vegetables",
    price: 38.87,
    unit: "/kg",
    change: 2.0,
  },
  {
    name: "Radish",
    localName: "Mula",
    category: "Root Vegetables",
    price: 42.87,
    unit: "/kg",
    change: -3.4,
  },
  {
    name: "Spinach",
    localName: "Palungo",
    category: "Leafy Greens",
    price: 47.31,
    unit: "/bunch",
    change: 2.8,
  },
  {
    name: "Tomato",
    localName: "Golbheda",
    category: "Nightshades",
    price: 58.0,
    unit: "/kg",
    change: -9.4,
  },
]

// Chart data for each vegetable
const chartData = {
  chart1: { data: [78, 82, 92, 88, 85, 80, 82, 90], positive: true },
  chart2: { data: [52, 55, 58, 54, 52, 58, 60, 50], positive: false },
  chart3: { data: [168, 172, 180, 175, 170, 165, 162, 165], positive: false },
  chart4: { data: [80, 78, 75, 78, 82, 85, 81, 77], positive: false },
  chart5: { data: [50, 48, 45, 48, 52, 46, 44, 43], positive: false },
  chart6: { data: [42, 40, 38, 36, 35, 38, 40, 39], positive: true },
  chart7: { data: [48, 46, 44, 42, 43, 44, 42, 43], positive: false },
  chart8: { data: [44, 42, 45, 50, 48, 46, 48, 47], positive: true },
  chart9: { data: [65, 64, 62, 60, 58, 56, 57, 58], positive: false },
}

// Initialize dashboard
function initDashboard() {
  const dashboard = document.getElementById("dashboard")

  vegetables.forEach((veg, index) => {
    const card = createPriceCard(veg, index)
    dashboard.appendChild(card)
  })
}

// Create price card
function createPriceCard(veg, index) {
  const card = document.createElement("div")
  card.className = "price-card"

  const changeClass = veg.change > 0 ? "positive" : "negative"
  const changeIcon = veg.change > 0 ? "↗" : "↘"
  const changeSymbol = veg.change > 0 ? "+" : ""

  card.innerHTML = `
        <div class="card-header">
            <div class="card-title">
                <h3>${veg.name}</h3>
                <p>${veg.localName}</p>
            </div>
            <span class="category-tag">${veg.category}</span>
        </div>
        <div class="price-info">
            <div>
                <span class="price">Rs. ${veg.price.toFixed(2)}</span>
                <span class="unit">${veg.unit}</span>
            </div>
            <span class="change ${changeClass}">
                ${changeIcon} ${changeSymbol}${veg.change}%
            </span>
        </div>
        <div class="chart-container">
            <canvas id="chart-${index + 1}"></canvas>
        </div>
    `

  return card
}

// Draw chart function
function drawChart(canvasId, data, isPositive) {
  const canvas = document.getElementById(canvasId)
  if (!canvas) return

  const ctx = canvas.getContext("2d")
  const rect = canvas.getBoundingClientRect()

  // Set canvas resolution
  const dpr = window.devicePixelRatio || 1
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.scale(dpr, dpr)

  const width = rect.width
  const height = rect.height
  const padding = 10

  // Calculate min and max
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  // Calculate points
  const points = data.map((value, i) => {
    const x = padding + (i / (data.length - 1)) * (width - padding * 2)
    const y = height - padding - ((value - min) / range) * (height - padding * 2)
    return { x, y }
  })

  // Colors based on positive/negative
  const lineColor = isPositive ? "#ef4444" : "#3b82f6"
  const fillColor1 = isPositive ? "rgba(239, 68, 68, 0.2)" : "rgba(59, 130, 246, 0.2)"
  const fillColor2 = isPositive ? "rgba(239, 68, 68, 0.05)" : "rgba(59, 130, 246, 0.05)"

  // Create gradient
  const gradient = ctx.createLinearGradient(0, 0, 0, height)
  gradient.addColorStop(0, fillColor1)
  gradient.addColorStop(1, fillColor2)

  // Draw filled area
  ctx.beginPath()
  ctx.moveTo(points[0].x, height - padding)
  ctx.lineTo(points[0].x, points[0].y)

  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y)
  }

  ctx.lineTo(points[points.length - 1].x, height - padding)
  ctx.closePath()
  ctx.fillStyle = gradient
  ctx.fill()

  // Draw line
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)

  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y)
  }

  ctx.strokeStyle = lineColor
  ctx.lineWidth = 2.5
  ctx.lineJoin = "round"
  ctx.lineCap = "round"
  ctx.stroke()
}

// Initialize all charts
function initCharts() {
  Object.keys(chartData).forEach((chartId) => {
    const { data, positive } = chartData[chartId]
    drawChart(chartId, data, positive)
  })
}

// Redraw on window resize
let resizeTimeout
window.addEventListener("resize", () => {
  clearTimeout(resizeTimeout)
  resizeTimeout = setTimeout(initCharts, 250)
})

// Initialize on load
window.addEventListener("DOMContentLoaded", initCharts)

// Theme toggle (placeholder functionality)
document.getElementById("themeToggle").addEventListener("click", () => {
  console.log("Theme toggle clicked")
})
