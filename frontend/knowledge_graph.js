/**
 * knowledge_graph.js — D3.js World Graph Visualization
 * Renders the wizarding world knowledge graph as an interactive force-directed graph.
 */

const KnowledgeGraph = (() => {
  let svg = null;
  let simulation = null;
  let graphData = { nodes: [], links: [] };
  let width = 0, height = 0;

  const NODE_COLORS = {
    location: '#4A90D9',
    npc: '#7ED321',
    quest: '#F5A623',
    item: '#BD10E0',
    faction: '#E74C3C',
    player: '#FFD700'
  };

  const NODE_SIZES = {
    location: 10,
    npc: 7,
    quest: 8,
    item: 5,
    faction: 9,
    player: 14
  };

  function init(containerId = 'world-graph-svg') {
    svg = d3.select(`#${containerId}`);
    const container = document.getElementById(containerId);
    if (!container) return;

    width = container.clientWidth || 220;
    height = container.clientHeight || 280;

    svg.attr('width', width).attr('height', height);

    // Add defs for glow filter
    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        svg.select('.graph-main-group').attr('transform', event.transform);
      });

    svg.call(zoom);
    svg.append('g').attr('class', 'graph-main-group');
  }

  function render(data) {
    if (!svg) return;
    graphData = data;

    const g = svg.select('.graph-main-group');
    g.selectAll('*').remove();

    if (!data.nodes || data.nodes.length === 0) {
      g.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#7A6A5A')
        .attr('font-size', '11px')
        .text('World graph loading...');
      return;
    }

    // Simulation
    simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links)
        .id(d => d.id)
        .distance(40)
        .strength(0.3))
      .force('charge', d3.forceManyBody().strength(-60))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(14))
      .alphaDecay(0.02);

    // Links
    const link = g.selectAll('.graph-link')
      .data(data.links)
      .enter().append('line')
      .attr('class', 'graph-link')
      .attr('stroke', 'rgba(212,175,55,0.15)')
      .attr('stroke-width', 1);

    // Node groups
    const node = g.selectAll('.graph-node')
      .data(data.nodes)
      .enter().append('g')
      .attr('class', 'graph-node')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragEnded));

    // Node circles
    node.append('circle')
      .attr('r', d => NODE_SIZES[d.type] || 7)
      .attr('fill', d => d.current ? '#FFD700' : (NODE_COLORS[d.type] || '#999'))
      .attr('stroke', d => d.current ? '#FFD700' : 'rgba(255,255,255,0.2)')
      .attr('stroke-width', d => d.current ? 3 : 1)
      .style('filter', d => d.current ? 'url(#glow)' : 'none');

    // Node labels (only for important nodes)
    node.filter(d => ['location', 'player'].includes(d.type))
      .append('text')
      .attr('dy', d => (NODE_SIZES[d.type] || 7) + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '8px')
      .attr('fill', '#B8A890')
      .text(d => d.label.length > 12 ? d.label.substring(0, 10) + '…' : d.label);

    // Tooltip
    node.append('title')
      .text(d => `${d.label}\n[${d.type}]`);

    // Pulse animation for player node
    node.filter(d => d.type === 'player')
      .append('circle')
      .attr('r', 14)
      .attr('fill', 'none')
      .attr('stroke', '#FFD700')
      .attr('stroke-width', 1)
      .style('opacity', 0.6)
      .append('animate')
        .attr('attributeName', 'r')
        .attr('values', '14;20;14')
        .attr('dur', '2s')
        .attr('repeatCount', 'indefinite');

    // Click handler
    node.on('click', (event, d) => {
      showNodeInfo(d);
    });

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => clamp(d.source.x, 10, width - 10))
        .attr('y1', d => clamp(d.source.y, 10, height - 10))
        .attr('x2', d => clamp(d.target.x, 10, width - 10))
        .attr('y2', d => clamp(d.target.y, 10, height - 10));

      node.attr('transform', d =>
        `translate(${clamp(d.x, 12, width - 12)},${clamp(d.y, 12, height - 12)})`
      );
    });
  }

  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  function dragStarted(event, d) {
    if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x; d.fy = event.y;
  }

  function dragEnded(event, d) {
    if (!event.active && simulation) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  function showNodeInfo(node) {
    // Show node info in a small tooltip in the corner
    const event = new CustomEvent('graph:nodeSelected', { detail: node });
    document.dispatchEvent(event);
  }

  function highlightNode(nodeId) {
    if (!svg) return;
    svg.selectAll('.graph-node circle')
      .style('filter', d => d.id === nodeId ? 'url(#glow) brightness(1.5)' : (d.current ? 'url(#glow)' : 'none'));
  }

  function updatePlayerLocation(locationId) {
    if (!svg || !graphData.nodes) return;
    graphData.nodes.forEach(n => { n.current = n.id === locationId; });
    svg.selectAll('.graph-node circle')
      .attr('fill', d => d.current ? '#FFD700' : (NODE_COLORS[d.type] || '#999'))
      .attr('r', d => d.current ? (NODE_SIZES[d.type] || 7) + 3 : (NODE_SIZES[d.type] || 7))
      .style('filter', d => d.current ? 'url(#glow)' : 'none');
  }

  // Build a simple mock graph for offline use
  function buildMockGraph() {
    return {
      nodes: [
        { id: 'player', label: 'You', type: 'player', current: true },
        { id: 'loc_001', label: 'Three Broomsticks', type: 'location', current: false },
        { id: 'loc_002', label: 'Hogwarts', type: 'location', current: false },
        { id: 'loc_003', label: 'Knockturn Alley', type: 'location', current: false },
        { id: 'loc_004', label: 'Forbidden Forest', type: 'location', current: false },
        { id: 'loc_005', label: 'Ministry', type: 'location', current: false },
        { id: 'npc_001', label: 'Rosmerta', type: 'npc' },
        { id: 'npc_002', label: 'Neville', type: 'npc' },
        { id: 'npc_003', label: 'McGonagall', type: 'npc' },
        { id: 'quest_001', label: 'Quest 1', type: 'quest' },
        { id: 'faction_001', label: 'Ministry', type: 'faction' },
        { id: 'faction_002', label: 'Order', type: 'faction' },
      ],
      links: [
        { source: 'player', target: 'loc_001', relationship: 'currently_at' },
        { source: 'loc_001', target: 'loc_002', relationship: 'connected_to' },
        { source: 'loc_001', target: 'loc_003', relationship: 'connected_to' },
        { source: 'loc_002', target: 'loc_004', relationship: 'connected_to' },
        { source: 'loc_002', target: 'loc_005', relationship: 'connected_to' },
        { source: 'npc_001', target: 'loc_001', relationship: 'located_in' },
        { source: 'npc_002', target: 'loc_001', relationship: 'located_in' },
        { source: 'npc_003', target: 'loc_002', relationship: 'located_in' },
        { source: 'npc_002', target: 'quest_001', relationship: 'gives_quest' },
        { source: 'faction_001', target: 'loc_005', relationship: 'controls' },
      ]
    };
  }

  return { init, render, updatePlayerLocation, highlightNode, buildMockGraph };
})();
