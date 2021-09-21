import QtQuick 2.15
import QtQuick.Window 2.2
import QtQuick.Controls 2.2

ApplicationWindow {
    id: root
    visible: true
    width: 640
    height: 480

    Connections {
        target: context

        function onResetEmitted() {
            console.log("game reset");
        }
        function onGameOver(winner) {
            console.log("game over, player " + winner + " won");
            newGameButton.visible = true
        }

        function onActivePlayerChanged() {
            console.log("Active player: " + context.activePlayer)
        }
    }

    TableView {
        id: table
        anchors.fill: parent
        model: context.board
        columnSpacing: 2
        rowSpacing: 2
        delegate: Button {
            implicitWidth: 100
            implicitHeight: 100
            text: tile.value
            enabled: !tile.filled
            highlighted: tile.winning
            onClicked: {
                context.doMove(column + table.columns * row)  // row, column are attached properties to the delegate
            }
        }
    }

    Button {
        id: newGameButton
        anchors {
            bottom: parent.bottom
            horizontalCenter: parent.horizontalCenter
        }
        visible: false
        text: "Play Again!"
        onClicked: {
            context.playAgain()
            newGameButton.visible = false
        }
    }
}
