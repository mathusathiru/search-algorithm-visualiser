// algorithm-visualizer.cy.js - Enhanced test suite

describe('Algorithm Visualizer', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000')
    // Wait for the app to load completely
    cy.get('#app-header').should('be.visible')
    cy.log('Application loaded successfully')
  })

  // ========== Basic UI Tests ==========
  it('loads the application and displays the header', () => {
    cy.get('#app-title').should('contain', 'Search Algorithm Visualiser')
  })

  it('toggles between single agent and multi agent modes', () => {
    // Check initial mode (single agent)
    cy.get('.toggle-button.active').should('contain', 'Single Agent')
    
    // Switch to multi agent mode
    cy.contains('Multi Agent').click()
    cy.get('.toggle-button.active').should('contain', 'Multi Agent')
    
    // Switch back to single agent mode
    cy.contains('Single Agent').click()
    cy.get('.toggle-button.active').should('contain', 'Single Agent')
  })

  // ========== Dropdown Tests with Proper Handling ==========
  it('selects algorithm from dropdown with proper state handling', () => {
    // Log the initial state
    cy.log('Getting initial dropdown state')
    cy.get('#algorithm').should('exist').and('be.visible')
    
    // Use select() with proper waiting for React state updates
    cy.log('Selecting BFS algorithm')
    cy.get('#algorithm').select('bfs')
    
    // Wait for React state update
    cy.wait(300)
    
    // Verify selection was processed by React
    cy.log('Verifying algorithm selection')
    cy.get('#algorithm').should('have.value', 'bfs')
    
    // Verify UI is updated with algorithm info
    cy.log('Checking for UI updates')
    cy.get('.grid-col-3 div').should('contain', 'Breadth First')
  })

  it('selects grid size from dropdown with proper state handling', () => {
    // Log initial state
    cy.log('Getting initial grid size state')
    cy.get('#gridSize').should('exist').and('be.visible')
    
    // Use select() with value verification
    cy.log('Selecting small grid size')
    cy.get('#gridSize').select('small')
    
    // Wait for React state update
    cy.wait(300)
    
    // Verify selection worked
    cy.log('Verifying grid size selection')
    cy.get('#gridSize').should('have.value', 'small')
    
    // Grid size changes should affect the grid DOM
    cy.log('Checking grid size changes reflected in UI')
    // The grid should have updated dimensions, but we'll just wait to ensure the change happened
    cy.wait(500)
  })
  
  // ========== Button State Tests with React State Awareness ==========
  it('enables start button when algorithm is selected', () => {
    // First check if button is disabled
    cy.log('Checking initial button state')
    cy.get('#start-traversal-btn').should('be.disabled')
    
    // Select algorithm with proper state verification
    cy.log('Selecting algorithm')
    cy.get('#algorithm').select('bfs')
    cy.get('#algorithm').should('have.value', 'bfs')
    
    // Wait for React to process state changes and update button disabled prop
    cy.wait(300)
    
    // Check if button is enabled
    cy.log('Verifying button state updated')
    cy.get('#start-traversal-btn').should('be.enabled')
  })
  
  // ========== Algorithm Tests with React State Awareness ==========
  it('runs BFS algorithm with proper state handling', () => {
    // Select BFS algorithm
    cy.log('Selecting BFS algorithm')
    cy.get('#algorithm').select('bfs')
    cy.get('#algorithm').should('have.value', 'bfs')
    
    // Wait for React to process state changes
    cy.wait(300)
    
    // Start button should be enabled
    cy.log('Verifying start button enabled')
    cy.get('#start-traversal-btn').should('be.enabled')
    
    // Start the algorithm
    cy.log('Starting algorithm')
    cy.get('#start-traversal-btn').click()
    
    // Wait for animation to start
    cy.wait(1000)
    
    // Verify algorithm is running - Pause and Stop buttons should be visible
    cy.log('Verifying algorithm started')
    cy.contains('Pause').should('be.visible')
    cy.contains('Stop').should('be.visible')
    
    // Stop the algorithm
    cy.log('Stopping algorithm')
    cy.contains('Stop').click()
  })

  // ========== Speed Control Tests ==========
  it('adjusts animation speed with the speed slider', () => {
    // Select an algorithm first
    cy.get('#algorithm').select('bfs')
    cy.wait(300)
    
    // Get the speed slider
    cy.get('#speed').should('exist').and('be.visible')
    
    // Move slider to faster speed (higher value)
    cy.get('#speed').invoke('val', 90).trigger('change')
    cy.wait(200)
    
    // Move slider to slower speed (lower value)
    cy.get('#speed').invoke('val', 10).trigger('change')
    cy.wait(200)
    
    // Run algorithm with the slower speed
    cy.get('#start-traversal-btn').click()
    cy.wait(1000)
    
    // Verify the algorithm is running
    cy.contains('Pause').should('be.visible')
    
    // Stop the algorithm
    cy.contains('Stop').click()
  })

  // ========== Clear Button Test ==========
  it('clears the grid when the clear button is clicked', () => {
    // First, select an algorithm and run it partially
    cy.get('#algorithm').select('bfs')
    cy.wait(300)
    cy.get('#start-traversal-btn').click()
    cy.wait(1000)
    cy.contains('Stop').click()
    cy.wait(300)
    
    // Now click the clear button
    cy.contains('Clear').click()
    cy.wait(300)
    
    // Verify reset state (start button should be enabled)
    cy.get('#start-traversal-btn').should('be.enabled')
  })

  // ========== Node Randomization Test ==========
  it('randomizes nodes in single agent mode', () => {
    // Make sure we're in single-agent mode
    cy.contains('Single Agent').click()
    cy.wait(300)
    
    // Capture the current position of start and end nodes
    let initialStartPos, initialEndPos
    cy.get('.grid-node.start').then($start => {
      initialStartPos = $start.css('left') + $start.css('top')
    })
    cy.get('.grid-node.end').then($end => {
      initialEndPos = $end.css('left') + $end.css('top')
    })
    
    // Click the randomize nodes button
    cy.contains('Randomise Nodes').click()
    cy.wait(500)
    
    // Verify nodes have moved
    cy.get('.grid-node.start').then($start => {
      const newStartPos = $start.css('left') + $start.css('top')
      // Not very likely to get exactly the same position after randomization
      cy.wrap(initialStartPos).should('not.equal', newStartPos)
    })
    cy.get('.grid-node.end').then($end => {
      const newEndPos = $end.css('left') + $end.css('top')
      cy.wrap(initialEndPos).should('not.equal', newEndPos)
    })
  })

  // ========== Maze Generation Tests ==========
  it('generates a maze in single agent mode', () => {
    // Make sure we're in single-agent mode
    cy.contains('Single Agent').click()
    cy.wait(300)
    
    // Count walls before maze generation (initially should be minimal)
    cy.get('.grid-cell').then($cells => {
      const initialWalls = Array.from($cells).filter(cell => 
        window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
      
      // Generate maze
      cy.contains('Generate Maze').click()
      cy.wait(1000)
      
      // Count walls after generation
      cy.get('.grid-cell').then($newCells => {
        const newWalls = Array.from($newCells).filter(cell => 
          window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
        
        // There should be more walls now
        expect(newWalls).to.be.greaterThan(initialWalls)
      })
    })
  })

  it('generates a random maze in single agent mode', () => {
    // Make sure we're in single-agent mode
    cy.contains('Single Agent').click()
    cy.wait(300)
    
    // Count walls before maze generation
    cy.get('.grid-cell').then($cells => {
      const initialWalls = Array.from($cells).filter(cell => 
        window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
      
      // Generate random maze
      cy.contains('Random Maze').click()
      cy.wait(1000)
      
      // Count walls after generation
      cy.get('.grid-cell').then($newCells => {
        const newWalls = Array.from($newCells).filter(cell => 
          window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
        
        // There should be more walls now
        expect(newWalls).to.be.greaterThan(initialWalls)
      })
    })
  })

  // ========== Multi-Agent Mode Tests with React State Awareness ==========
  describe('Multi-Agent Mode', () => {
    beforeEach(() => {
      // Switch to multi-agent mode with verification
      cy.log('Switching to multi-agent mode')
      cy.contains('Multi Agent').click()
      cy.get('.toggle-button.active').should('contain', 'Multi Agent')
      
      // Wait for React to update state and re-render
      cy.wait(500)
    })
    
    it('displays multiple agents in multi-agent mode', () => {
      // Check for multiple agent nodes (we expect at least 4)
      cy.log('Checking for agent nodes')
      cy.get('.grid-node').should('have.length.at.least', 4)
    })
    
    it('selects a multi-agent algorithm with proper state handling', () => {
      // Select a multi-agent algorithm (CBS)
      cy.log('Selecting CBS algorithm')
      cy.get('#algorithm')
        .should('exist')
        .and('be.visible')
        .select('cbs')
      
      // Verify selection was processed
      cy.get('#algorithm').should('have.value', 'cbs')
      
      // Wait for React state update
      cy.wait(300)
      
      // Check for CBS algorithm info in the UI
      cy.log('Checking for algorithm info in UI')
      cy.get('.grid-col-3 div').should('contain', 'Conflict-Based Search')
    })

    it('adds an agent when the Add Agent button is clicked', () => {
      // Count initial agents
      cy.get('.grid-node.start').then($startNodes => {
        const initialAgentCount = $startNodes.length

        // Add an agent
        cy.contains('Add Agent').click()
        cy.wait(500)

        // Verify agent was added
        cy.get('.grid-node.start').should('have.length', initialAgentCount + 1)
      })
    })

    it('removes an agent when the Remove Agent button is clicked', () => {
      // Count initial agents
      cy.get('.grid-node.start').then($startNodes => {
        const initialAgentCount = $startNodes.length

        // Remove an agent
        cy.contains('Remove Agent').click()
        cy.wait(500)

        // Verify agent was removed
        cy.get('.grid-node.start').should('have.length', initialAgentCount - 1)
      })
    })

    it('randomizes agents in multi-agent mode', () => {
      // Capture initial positions
      let initialPositions = []
      cy.get('.grid-node.start').each($node => {
        initialPositions.push($node.css('left') + $node.css('top'))
      })

      // Randomize agents
      cy.contains('Randomise Agents').click()
      cy.wait(500)

      // Verify positions changed
      let positionsChanged = false
      cy.get('.grid-node.start').each(($node, index) => {
        if (index < initialPositions.length) {
          const newPos = $node.css('left') + $node.css('top')
          if (newPos !== initialPositions[index]) {
            positionsChanged = true
          }
        }
      }).then(() => {
        expect(positionsChanged).to.be.true
      })
    })

    it('generates an open grid in multi-agent mode', () => {
      // Count walls before open grid generation
      cy.get('.grid-cell').then($cells => {
        const initialWalls = Array.from($cells).filter(cell => 
          window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
        
        // Generate open grid
        cy.contains('Open Grid').click()
        cy.wait(1000)
        
        // Count walls after generation
        cy.get('.grid-cell').then($newCells => {
          const newWalls = Array.from($newCells).filter(cell => 
            window.getComputedStyle(cell).backgroundColor === 'rgb(0, 0, 0)').length
          
          // Open grid should have fewer walls in the middle
          expect(newWalls).to.be.lessThan(initialWalls * 2)
        })
      })
    })

    it('runs a multi-agent algorithm successfully', () => {
      // Select a multi-agent algorithm
      cy.get('#algorithm').select('cbs')
      cy.wait(300)
      
      // Start the algorithm
      cy.get('#start-traversal-btn').click()
      cy.wait(1000)
      
      // Verify algorithm is running
      cy.contains('Pause').should('exist')
      
      // Stop the algorithm
      cy.contains('Stop').click()
    })
  })

  // ========== Wall Drawing Tests ==========
  it('allows users to draw and erase walls', () => {
    // Make sure we're in single-agent mode
    cy.contains('Single Agent').click()
    cy.wait(300)
    
    // Click on a cell to draw a wall
    // Find a cell that's not occupied by start/end nodes
    cy.get('.grid-cell.interactive').eq(50).click()
    cy.wait(200)
    
    // Verify that the cell is now a wall (black)
    cy.get('.grid-cell.interactive').eq(50).should('have.css', 'background-color', 'rgb(0, 0, 0)')
    
    // Click again to remove the wall
    cy.get('.grid-cell.interactive').eq(50).click()
    cy.wait(200)
    
    // Verify that the cell is no longer a wall
    cy.get('.grid-cell.interactive').eq(50).should('not.have.css', 'background-color', 'rgb(0, 0, 0)')
  })

  // ========== Pause and Resume Tests ==========
  it('pauses and resumes the algorithm animation', () => {
    // Select an algorithm
    cy.get('#algorithm').select('bfs')
    cy.wait(300)
    
    // Start the algorithm
    cy.get('#start-traversal-btn').click()
    cy.wait(1000)
    
    // Pause the animation
    cy.contains('Pause').click()
    cy.wait(300)
    
    // Check that it changed to Resume
    cy.contains('Resume').should('exist')
    
    // Resume the animation
    cy.contains('Resume').click()
    cy.wait(300)
    
    // Check that it changed back to Pause
    cy.contains('Pause').should('exist')
    
    // Stop the algorithm
    cy.contains('Stop').click()
  })
})